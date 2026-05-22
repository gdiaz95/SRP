""" dt_net_2d.py
    DeepThinking network 2D.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn
import torch.nn.functional as F
import time
from torchvision.utils import save_image

from .blocks import BasicBlock2D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, group_norm=False, **kwargs):
        super().__init__()

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)
    

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):

        if interim_thought is None:
            initial_thought = self.projection(x)
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        # out=self.head(interim_thought)

        if self.training:
            return out, interim_thought
        

        return all_outputs,interim_thought# return out,interim_thought #
    
    def forward_parallel(self, x, iters_to_do, numchunks, overlap, device, interims=None, **kwargs):

        width=self.width
        step_per_chunk=1
        initial=False
        total_overlap = (numchunks - 1) * overlap
        height=x.shape[2]
        chunk_height = (height + total_overlap) // numchunks
        outputs = torch.zeros(numchunks, x.shape[0],step_per_chunk,2, chunk_height, x.shape[3],device=x.device)
        if interims==None:
            interims=torch.zeros(numchunks, x.shape[0],width, chunk_height, x.shape[3],device=x.device)
            initial=True

        chunks=split_image_into_chunks(x, numchunks, overlap)
       
        gpu_count = torch.cuda.device_count()
        for j in range(iters_to_do):

            
            futures = []

            for i in range(numchunks):
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                chunk = chunks[i].to(chunk_device)
                interim = interims[i].to(chunk_device)
                self.to(chunk_device)

                futures.append(
                    torch.jit.fork(self.special_forward, chunk, step_per_chunk, initial,interim)
                )

            for i,future in enumerate (futures):
                output, interim = torch.jit.wait(future)
                outputs[i] = output
                interims[i] = interim

            all_outputs = combine_chunks(outputs, numchunks, overlap)
            # save_image(all_outputs[0, -1, 1], f"/home/gabriel/output_{j}.png")
            
            initial=False
        
            single_image = apply_mask(x, all_outputs)
            # save_image(single_image[0], f"/home/gabriel/single_image_{j}.png")

        return single_image,interims

    def special_forward(self,x, iters_to_do, initial,interim_thought=None, **kwargs):
        if initial:
            initial_thought = self.projection(x)
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        return all_outputs,interim_thought


def apply_mask(inputs, all_outputs):
    channel_sum = torch.sum(inputs, dim=1)
    just_path_single = (channel_sum > 0).float().unsqueeze(1).expand(-1, 3, -1, -1)
    single_predicted = all_outputs[:, -1, 1]
    reshape_single = single_predicted.view(inputs.size(0), inputs.shape[2], inputs.shape[3]).unsqueeze(1).expand(-1, 3, -1, -1)
    overlayed_maze_single = reshape_single * inputs
    green_mask = (inputs[:, 1, :, :] == 1) & (inputs[:, 0, :, :] == 0) & (inputs[:, 2, :, :] == 0)
    overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
    overlayed_maze_single = overlayed_maze_single * inputs
    a_single = torch.clamp(overlayed_maze_single + just_path_single * 0.2, 0, 1)
    return a_single

def combine_chunks(outputs, num_chunks,overlap, elimination_height=1):
    num_chunks = outputs.shape[0]
    batch_size, iterations, _, chunk_height, width = outputs[0].shape
    stride = chunk_height - overlap

    if overlap >= 3:
        elimination_height = 1
    else:
        elimination_height = 0

    # Create a tensor to hold the combined output
    output_height = stride * (num_chunks - 1) + chunk_height
    output_grid = torch.zeros((batch_size, iterations, 2, output_height, width), device=outputs[0].device)

    for i in range(num_chunks):
        chunk = outputs[i]

        # Set the elimination_height rows to zeros
        if elimination_height > 0:
            chunk[:, :, :, :elimination_height, :] = 0
            chunk[:, :, :, -elimination_height:, :] = 0

        start_idx = i * stride
        # end_idx = start_idx + chunk_height

        output_grid[:, :, :, start_idx:start_idx+chunk_height, :]=torch.max(output_grid[:, :, :, start_idx:start_idx+chunk_height, :] , chunk)

    return output_grid

def split_image_into_chunks(images, num_chunks, overlap=0):
    """
    Splits the image into num_chunks^2 chunks along both width and height.

    Parameters:
    image (Tensor): The input image tensor of shape (batch_size, channels, height, width).
    num_chunks (int): The number of chunks per dimension.

    Returns:
    Tensor: A tensor of concatenated image chunks.
    """
    batch_size, channels, height, width = images.shape
    # Calculate the height of each chunk, accounting for the overlap
    total_overlap = (num_chunks - 1) * overlap
    chunk_height = (height + total_overlap) // num_chunks

    # Create an empty tensor to hold the concatenated chunks
    concatenated_chunks = torch.empty(num_chunks,batch_size, channels, chunk_height, width, device=images.device)

    for i in range(num_chunks):
        
        # Calculate the indices where the current chunk should be placed in the image
        start_index = i * (chunk_height - overlap)
        end_index = start_index + chunk_height

        # Crop the chunk from the image and place it into the concatenated tensor
        concatenated_chunks[i,:, :, 0:chunk_height, :] = images[:, :, start_index:end_index, :]

    return concatenated_chunks

def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recall_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)


def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)