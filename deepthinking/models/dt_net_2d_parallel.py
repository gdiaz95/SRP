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
import numpy as np
import torch.multiprocessing as mp
# from multiprocessing import Process, Queue
# import multiprocessing


# Set the seed for reproducibility
# import random
# import numpy as np
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# if not multiprocessing.get_start_method(allow_none=True):
#     multiprocessing.set_start_method('spawn')


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

        self.conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3, #added the self
                                stride=1, padding=1, bias=False)
        
        # self.conv_dynamic=nn.Conv2d(2*width, width, kernel_size=3, #added the self
        #                         stride=1, padding=1, bias=False)

        recur_layers = []
        # if recall:
        #     recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False) #added the self

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        # self.recur_block_0 = nn.Sequential(conv_recall)
        for i in range(len(recur_layers[0])):
            setattr(self, f'recur_block_{i}', recur_layers[0][i])
        # self.recur_block = nn.Sequential()
        self.head1 = nn.Sequential(head_conv1, nn.ReLU())
        self.head2 = nn.Sequential(head_conv2, nn.ReLU())
        # self.head3 = nn.Sequential(head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return layers
    
    def forward_multichannel(self, x, iters_to_do, numchunks, overlap, device, chunks_interims=None):

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # recall_correct = torch.load('/home/gabriel/tensors/recall.pt')
        # conv_0_1_correct = torch.load('/home/gabriel/tensors/conv_0_1.pt')
        # conv_0_2_correct = torch.load('/home/gabriel/tensors/conv_0_2.pt')
        # conv_1_1_correct = torch.load('/home/gabriel/tensors/conv_1_1.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head1_correct = torch.load('/home/gabriel/tensors/head1.pt')
        # head2_correct = torch.load('/home/gabriel/tensors/head2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')
        start_time_total = time.time()
        width=self.width
        total_overlap = (numchunks - 1) * overlap
        height=x.shape[2]
        batch_size=x.shape[0]
        chunk_height = (height + total_overlap) // numchunks
        # outputs_conv_head3 = torch.zeros(numchunks, x.shape[0], 2, chunk_height, x.shape[3],device=x.device)
        # outputs_conv= torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        chunks=split_image_into_chunks(x, numchunks, overlap)
        numchunks=chunks.shape[0]
        chunks= chunks.view(numchunks*chunks.shape[1],chunks.shape[2], chunk_height, chunks.shape[4])

        
        # are_equal = torch.equal(x_correct, x)
    
        if chunks_interims is None:
            # line_time = time.time()
            initial_thought = self.projection(x)
            # print(f"Time for projection: {time.time() - line_time:.5f} seconds")
            # torch.save(initial_thought,'/home/gabriel/tensors/projection.pt')
            # are_equal = torch.equal(initial_thought, initial_thought_correct)
            chunks_interims = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # are_equal=torch.equal(interim_thought, initial_thought_correct)

        for i in range(iters_to_do):

            chunks_interims = split_image_into_chunks(chunks_interims, numchunks, overlap)
            chunks_interims = chunks_interims.view(numchunks*batch_size,width, chunk_height, chunks_interims.shape[4])

            if self.recall:
                chunks_interims = torch.cat([chunks_interims, chunks], 1)
            # line_time = time.time()
            chunks_interims = self.conv_recall(chunks_interims)
            # chunks_interims=chunks_interims.view(numchunks,batch_size,width,chunk_height,chunks_interims.shape[3])
            # interims=combine_chunks(chunks_interims, numchunks, overlap, width)
            # print(f"Time for recall: {time.time() - line_time:.5f} seconds")
            # torch.save(chunks_interims, '/home/gabriel/tensors/recall.pt')
            # are_equal = torch.equal(interims, recall_correct)
            # if not are_equal:
            #     diff = torch.abs(interims- conv_0_2_correct)
            #     max_diff = torch.max(diff)
            # line_time = time.time()
            chunks_interims = self.recur_block_0.forward_pieces1(chunks_interims)
            # print(f"Time for conv 1: {time.time() - line_time:.5f} seconds")
            # torch.save(chunks_interims, '/home/gabriel/tensors/conv_0_1.pt')
            # are_equal = torch.equal(chunks_interims, conv_0_1_correct)
            # line_time = time.time()
            chunks_interims = self.recur_block_0.forward_pieces2(chunks_interims)
            # print(f"Time for conv 2: {time.time() - line_time:.5f} seconds")
            # torch.save(chunks_interims, '/home/gabriel/tensors/conv_0_2.pt')
            # are_equal = torch.equal(chunks_interims, conv_0_2_correct)
            # line_time = time.time()
            chunks_interims = self.recur_block_1.forward_pieces1(chunks_interims)
            # print(f"Time for conv 3: {time.time() - line_time:.5f} seconds")
            # torch.save(chunks_interims, '/home/gabriel/tensors/conv_1_1.pt')
            # are_equal = torch.equal(chunks_interims, conv_1_1_correct)
            # line_time = time.time()
            chunks_interims = self.recur_block_1.forward_pieces2(chunks_interims)
            # print(f"Time for conv 4: {time.time() - line_time:.5f} seconds")
            # torch.save(chunks_interims, '/home/gabriel/tensors/conv_1_2.pt')
            # are_equal = torch.equal(chunks_interims, conv_1_2_correct)
            # line_time = time.time()
            out = self.head1(chunks_interims)
            # print(f"Time for conv 5: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head1.pt')
            # are_equal = torch.equal(out, head1_correct)
            # line_time = time.time()
            out = self.head2(out)
            # print(f"Time for conv 6: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head2.pt')
            # are_equal = torch.equal(out, head2_correct)
            # line_time = time.time()
            out = self.head_conv3(out)
            # print(f"Time for conv 7: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head3.pt')
            # are_equal = torch.equal(out, head3_correct)

            chunks_interims=chunks_interims.view(numchunks,batch_size,width,chunk_height,chunks_interims.shape[3])
            chunks_interims=combine_chunks(chunks_interims, numchunks, overlap, width,cancelation_height=8)
            out=out.view(numchunks,batch_size,2,chunk_height,chunks_interims.shape[3])
            out=combine_chunks(out, numchunks, overlap, 2,cancelation_height=8)
            # are_equal = torch.equal(out, head3_correct)
            # if not are_equal:
            #     max_diff = torch.max(torch.abs(out-head3_correct))

            # are_equal = torch.equal(chunks_interims, conv_1_2_correct)
            # if not are_equal:
            #     max_diff = torch.max(torch.abs(chunks_interims-conv_1_2_correct))
            all_outputs[:, i] = out

        # out=self.head(chunks_interims)

        # if self.training:
        #     return out, chunks_interims
        
        single_image = apply_mask(x, all_outputs, iters_to_do-1)
        # print(f"Time for apply_mask: {time.time() - line_time:.5f} seconds")
            # save_image(single_image[0], f"/home/gabriel/single_image_{j}.png")
            # save_image(single_image[0], f"/home/gabriel/single_image_{j}.png")

        end_time=time.time()
        time_elapsed=(end_time-start_time_total)/10
        # np.save('/home/gabriel/times_forward/1K_5_chunks_1_splits_MC_1GPU.npy',time_elapsed/1000)
        return single_image,chunks_interims

    def forward(self, x, iters_to_do, interim_thought=None, update=False,**kwargs):

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # recall_correct = torch.load('/home/gabriel/tensors/recall.pt')
        # conv_0_1_correct = torch.load('/home/gabriel/tensors/conv_0_1.pt')
        # conv_0_2_correct = torch.load('/home/gabriel/tensors/conv_0_2.pt')
        # conv_1_1_correct = torch.load('/home/gabriel/tensors/conv_1_1.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head1_correct = torch.load('/home/gabriel/tensors/head1.pt')
        # head2_correct = torch.load('/home/gabriel/tensors/head2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')

        # torch.save(x, '/home/gabriel/tensors/x.pt')
        # are_equal = torch.equal(x_correct, x)

        if interim_thought is None:
            # line_time = time.time()
            initial_thought = self.projection(x)
            # print(f"Time for projection: {time.time() - line_time:.5f} seconds")
            # torch.save(initial_thought,'/home/gabriel/tensors/projection.pt')
            # are_equal = torch.equal(initial_thought, initial_thought_correct)
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            # line_time = time.time()
            interim_thought = self.conv_recall(interim_thought)
            # print(f"Time for recall: {time.time() - line_time:.5f} seconds")
            # torch.save(interim_thought, '/home/gabriel/tensors/recall.pt')
            # are_equal = torch.equal(interim_thought, recall_correct)
            # line_time = time.time()
            interim_thought = self.recur_block_0.forward_pieces1(interim_thought)
            # print(f"Time for conv 1: {time.time() - line_time:.5f} seconds")
            # torch.save(interim_thought, '/home/gabriel/tensors/conv_0_1.pt')
            # are_equal = torch.equal(interim_thought, conv_0_1_correct)
            # line_time = time.time()
            interim_thought = self.recur_block_0.forward_pieces2(interim_thought)
            # print(f"Time for conv 2: {time.time() - line_time:.5f} seconds")
            # torch.save(interim_thought, '/home/gabriel/tensors/conv_0_2.pt')
            # are_equal = torch.equal(interim_thought, conv_0_2_correct)
            # line_time = time.time()
            interim_thought = self.recur_block_1.forward_pieces1(interim_thought)
            # print(f"Time for conv 3: {time.time() - line_time:.5f} seconds")
            # torch.save(interim_thought, '/home/gabriel/tensors/conv_1_1.pt')
            # are_equal = torch.equal(interim_thought, conv_1_1_correct)
            # line_time = time.time()
            interim_thought = self.recur_block_1.forward_pieces2(interim_thought)
            # print(f"Time for conv 4: {time.time() - line_time:.5f} seconds")
            # torch.save(interim_thought, '/home/gabriel/tensors/conv_1_2.pt')
            # are_equal = torch.equal(interim_thought, conv_1_2_correct)
            # line_time = time.time()
            out = self.head1(interim_thought)
            # print(f"Time for conv 5: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head1.pt')
            # are_equal = torch.equal(out, head1_correct)
            # line_time = time.time()
            out = self.head2(out)
            # print(f"Time for conv 6: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head2.pt')
            # are_equal = torch.equal(out, head2_correct)
            # line_time = time.time()
            out = self.head_conv3(out)
            # print(f"Time for conv 7: {time.time() - line_time:.5f} seconds")
            # torch.save(out, '/home/gabriel/tensors/head3.pt')
            # are_equal = torch.equal(out, head3_correct)
            all_outputs[:, i] = out

        # out=self.head(interim_thought)

        if self.training:
            return out, interim_thought
        

        return all_outputs,interim_thought
    
    def forward_parallel(self, x, iters_to_do, numchunks, overlap, device, interims=None, **kwargs):

        width=self.width
        step_per_chunk=1
        initial=False
        total_overlap = (numchunks - 1) * overlap
        height=x.shape[2]
        chunk_height = (height + total_overlap) // numchunks
        gpu_count = torch.cuda.device_count()
        all_outputs = torch.zeros(x.shape[0],iters_to_do,2, x.shape[3], x.shape[3],device=x.device)
        # outputs_conv_0 = torch.zeros(numchunks, x.shape[0], width,chunk_height, x.shape[3],device=x.device) 
        # outputs_conv_recall = torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        # outputs_conv_1 = torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device) 
        # outputs_conv_2 = torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        # outputs_conv_3 = torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device) 
        # outputs_conv_4 = torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        outputs_conv= torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head1 = torch.zeros(numchunks, x.shape[0], 32, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head2 = torch.zeros(numchunks, x.shape[0], 8, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head3 = torch.zeros(numchunks, x.shape[0], 2, chunk_height, x.shape[3],device=x.device)
        chunks=split_image_into_chunks(x, numchunks, overlap)
        numchunks=chunks.shape[0]

        # print("Warming up the convolutions")
        self.warm_up_all_convolutions(numchunks, gpu_count, device, chunks,outputs_conv_head1,outputs_conv_head2,outputs_conv)

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # recall_correct = torch.load('/home/gabriel/tensors/recall.pt')
        # conv_0_1_correct = torch.load('/home/gabriel/tensors/conv_0_1.pt')
        # conv_0_2_correct = torch.load('/home/gabriel/tensors/conv_0_2.pt')
        # conv_1_1_correct = torch.load('/home/gabriel/tensors/conv_1_1.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head1_correct = torch.load('/home/gabriel/tensors/head1.pt')
        # head2_correct = torch.load('/home/gabriel/tensors/head2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')
       
        
        # start_time_total = time.time()
        # are_equal = torch.equal(x_correct, x)

        if interims==None:
        #     #################################################conv 0 projection 
            futures = [None] * numchunks

            # start_time = time.time()
            # print("time for proyection")
            # Process each chunk asynchronously
            for i in range(numchunks):
                # line_time = time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                
                # line_time = time.time()
                chunk = chunks[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line_time:.5f} seconds")
                
                # line_time = time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                
                # line_time = time.time()
                futures[i] = torch.jit.fork(self.special_forward_proj, chunk)
                # print(f"Time for line 4: {time.time() - line_time:.5f} seconds")

            # Collect results
            for i in range(numchunks):
                # line_time = time.time()
                outputs_conv[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")

            # Print the total execution time
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
                

            # for i in range(numchunks):
            #     chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
            #     chunk = chunks[i].to(chunk_device)
            #     self.to(chunk_device)
            #     torch.jit.fork(self.special_forward_proj, chunk)

            # futures = [None] * numchunks

            # # Record the start time
            # start_time = time.time()

            # for i in range(numchunks):
            #     line_time = time.time()
            #     chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
            #     print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                
            #     line_time = time.time()
            #     chunk = chunks[i].to(chunk_device)
            #     print(f"Time for line 2: {time.time() - line_time:.5f} seconds")
                
            #     line_time = time.time()
            #     self.to(chunk_device)
            #     print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                
            #     line_time = time.time()
            #     futures[i]=torch.jit.fork(self.special_forward_proj, chunk)
            #     print(f"Time for line 4: {time.time() - line_time:.5f} seconds")

            # for i in range (numchunks):
            #     line_time = time.time()
            #     output0 = torch.jit.wait(futures[i])
            #     print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                
            #     line_time = time.time()
            #     outputs_conv[i] = output0
            #     print(f"Time for line 6: {time.time() - line_time:.5f} seconds")can

            # # Print the total execution time
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")


            #############################################################################
            initial=True
            # line_time = time.time()
            interims0=combine_chunks(outputs_conv, numchunks, overlap, width)
            # print(f"Time for combine_chunks: {time.time() - line_time:.5f} seconds")
            hello=1
            # are_equal = torch.equal(interims0, initial_thought_correct)
            # if not are_equal:
            #     diff = torch.abs(interims0- initial_thought_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            # line_time = time.time()
            chunks0=split_image_into_chunks(interims0, numchunks, overlap)
            # print(f"Time for split_image_into_chunks: {time.time() - line_time:.5f} seconds")
            hello=2
        
        # print("printing time for next convolution")

        for j in range(iters_to_do):

            if self.recall:

                if not initial:
                    interims0=interims
                    chunks0=split_image_into_chunks(interims0, numchunks, overlap)

                #################################################conv 1 recur_block_recall
                # print("time for recall")
                # start_time = time.time()
                futures = [None]*numchunks
                for i in range(numchunks):
                    # line_time=time.time()
                    chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                    # print(f"Time for recall line 1: {time.time() - line_time:.5f} seconds")
                    # line_time=time.time()
                    chunk = chunks0[i].to(chunk_device)
                    # print(f"Time for recall line 2: {time.time() - line_time:.5f} seconds")
                    # line_time=time.time()
                    chunk_recall=chunks[i].to(chunk_device)
                    # print(f"Time for recall line 3: {time.time() - line_time:.5f} seconds")
                    # line_time=time.time()
                    self.to(chunk_device)
                    # print(f"Time for recall line 4: {time.time() - line_time:.5f} seconds")
                    # line_time=time.time()
                    futures[i]=torch.jit.fork(self.special_forward_conv_recall,chunk_recall,chunk)
                    # print(f"Time for recall line 5: {time.time() - line_time:.5f} seconds")
                    

                for i in range (numchunks):
                    # line_time=time.time()
                    outputs_conv[i] = torch.jit.wait(futures[i])
                    # print(f"Time for recall line 6: {time.time() - line_time:.5f} seconds")
                     

                # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
                # futures = []
                # start_time = time.time()
                # for i in range(numchunks):
                #     line_time = time.time()
                #     chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                #     print(f"Time for recall line 1: {time.time() - line_time:.5f} seconds")
                    
                #     line_time = time.time()
                #     chunk = chunks0[i].to(chunk_device)
                #     print(f"Time for recall line 2: {time.time() - line_time:.5f} seconds")
                    
                #     line_time = time.time()
                #     chunk_recall = chunks[i].to(chunk_device)
                #     print(f"Time for recall line 3: {time.time() - line_time:.5f} seconds")
                    
                #     line_time = time.time()
                #     self.to(chunk_device)
                #     print(f"Time for recall line 4: {time.time() - line_time:.5f} seconds")
                    
                #     line_time = time.time()
                #     futures.append(torch.jit.fork(self.special_forward_conv_recall, chunk_recall, chunk))
                #     print(f"Time for recall line 5: {time.time() - line_time:.5f} seconds")

                # for i, future in enumerate(futures):
                #     line_time = time.time()
                #     output0 = torch.jit.wait(future)
                #     print(f"Time for recall line 6: {time.time() - line_time:.5f} seconds")
                    
                #     line_time = time.time()
                #     outputs_conv[i] = output0
                #     print(f"Time for recall line 7: {time.time() - line_time:.5f} seconds")

                # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
                # for i in range (numchunks):
                #     chunk=chunks0[i]
                #     chunk_recall=chunks[i]
                #     outputs_conv_recall[i]= self.conv_recall(torch.cat([chunk, chunk_recall], 1))

            


                #############################################################################
                interims1=combine_chunks(outputs_conv, numchunks, overlap, width)
                # are_equal = torch.equal(interims1, recall_correct)
                # if not are_equal:
                #     diff = torch.abs(interims1- recall_correct)
                #     max_diff = torch.max(diff)

                #print(f"Tensors are equal: {are_equal}")
                chunks1=split_image_into_chunks(interims1, numchunks, overlap)

            #################################################conv 2 recur_block_0_1
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 2")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks1[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_recur_block_0_1, chunk)
                # print(f"Time for line 4: {time.time() - line_time:.5f} seconds")


            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                 
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range (numchunks):
            #     chunk=chunks1[i]
            #     outputs_conv_1[i] = self.recur_block_0.forward_pieces1(chunk)


            #############################################################################
            interims2=combine_chunks(outputs_conv, numchunks, overlap, width)
            # are_equal = torch.equal(interims2, conv_0_1_correct)
            # if not are_equal:
            #     diff = torch.abs(interims2- conv_0_1_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            chunks2=split_image_into_chunks(interims2, numchunks, overlap)

            #################################################conv 3 recur_block_0_2 
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 3")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks2[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # line_time=time.time()
                # print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                futures[i] = torch.jit.fork(self.special_forward_recur_block_0_2, chunk)
                # print(f"Time for line 4: {time.time() - line_time:.5f} seconds")
                

            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                 
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range (numchunks):
            #     chunk=chunks2[i]
            #     outputs_conv_2[i] = self.recur_block_0.forward_pieces2(chunk)

            
            #############################################################################
            interims3=combine_chunks(outputs_conv, numchunks, overlap, width)
            # are_equal = torch.equal(interims3, conv_0_2_correct)
            # if not are_equal:
            #     diff = torch.abs(interims3- conv_0_2_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            chunks3=split_image_into_chunks(interims3, numchunks, overlap)

            #################################################conv 4 recur_block_1_1
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 4")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks3[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_recur_block_1_1, chunk)
                # print(f"Time for line 4: {time.time() - line_time:.5f} seconds")


            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv[i]  = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range (numchunks):
            #     chunk=chunks3[i]
            #     outputs_conv_3[i] = self.recur_block_1.forward_pieces1(chunk)


            #############################################################################
            interims4=combine_chunks(outputs_conv, numchunks, overlap, width)
            # are_equal = torch.equal(interims4, conv_1_1_correct)
            # if not are_equal:
            #     diff = torch.abs(interims4- conv_1_1_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            chunks4=split_image_into_chunks(interims4, numchunks, overlap)

            #################################################conv 5 recur_block_1_2
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 5")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks4[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_recur_block_1_2, chunk)
                # print(f"Time for line 4: {time.time() - line_time:.5f} seconds")

            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                 
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range (numchunks):
            #     chunk=chunks4[i]
            #     outputs_conv_4[i] = self.recur_block_1.forward_pieces2(chunk)


            #############################################################################
            interims5=combine_chunks(outputs_conv, numchunks, overlap, width)
            # are_equal = torch.equal(interims5, conv_1_2_correct)
            # if not are_equal:
            #     diff = torch.abs(interims5 - conv_1_2_correct)
            #     max_diff = torch.max(diff)

            interims=interims5

            #print(f"Tensors are equal: {are_equal}")
            chunks5=split_image_into_chunks(interims5, numchunks, overlap)

            #################################################conv 6 head 1
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 6")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks5[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_head1, chunk)
                # print(f"Time for line 4: {time.time() - line.time:.5f} seconds")

            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv_head1[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range(numchunks):
            #     chunk=chunks5[i]
            #     outputs_conv_head1[i] = self.head1(chunk)

            
            #############################################################################
            interims6=combine_chunks(outputs_conv_head1, numchunks, overlap, 32)
            # are_equal = torch.equal(interims6, head1_correct)
            # if not are_equal:
            #     diff = torch.abs(interims6 - head1_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            chunks6=split_image_into_chunks(interims6, numchunks, overlap)

            ################################################# conv 7 head 2
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 7")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks6[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_head2, chunk)
                # print(f"Time for line 4: {time.time() - line.time:.5f} seconds")

            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv_head2[i]  = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line_time:.5f} seconds")
                
            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")
            # for i in range(numchunks):
            #     chunk=chunks6[i]
            #     outputs_conv_head2[i] = self.head2(chunk)

            #############################################################################     
            interims7=combine_chunks(outputs_conv_head2, numchunks, overlap, 8)
            # are_equal = torch.equal(interims7, head2_correct)
            # if not are_equal:
            #     diff = torch.abs(interims7 - head2_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")
            chunks7=split_image_into_chunks(interims7, numchunks, overlap)

            #################################################conv 8 head 3
            # start_time = time.time()
            futures = [None]*numchunks
            # print("time for conv 8")
            for i in range(numchunks):
                # line_time=time.time()
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                # print(f"Time for line 1: {time.time() - line_time:.5f} seconds")
                # line_time=time.time()
                chunk = chunks7[i].to(chunk_device)
                # print(f"Time for line 2: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                self.to(chunk_device)
                # print(f"Time for line 3: {time.time() - line.time:.5f} seconds")
                # line_time=time.time()
                futures[i]=torch.jit.fork(self.special_forward_head3, chunk)
                # print(f"Time for line 4: {time.time() - line.time:.5f} seconds")

            for i in range (numchunks):
                # line_time=time.time()
                outputs_conv_head3[i] = torch.jit.wait(futures[i])
                # print(f"Time for line 5: {time.time() - line.time:.5f} seconds")

                

            # for i in range(numchunks):
            #     chunk=chunks7[i]
            #     outputs_conv_head3[i] = self.head_conv3(chunk)

            # print(f"Total execution time: {time.time() - start_time:.5f} seconds")            
            #############################################################################
            outputs=combine_chunks(outputs_conv_head3, numchunks, overlap, 2)
            # are_equal = torch.equal(outputs, head3_correct)
            # if not are_equal:
            #     diff = torch.abs(outputs - head3_correct)
            #     max_diff = torch.max(diff)

            #print(f"Tensors are equal: {are_equal}")

            all_outputs[:,j]=outputs
            # save_image(all_outputs[0, j, 1], f"/home/gabriel/output.png")
            
            # save_image(all_outputs[0, -1, 1], f"/home/gabriel/output_{j}.png")
            
            initial=False
        
        # line_time = time.time()
        single_image = apply_mask(x, all_outputs, iters_to_do-1)
        # print(f"Time for apply_mask: {time.time() - line_time:.5f} seconds")
            # save_image(single_image[0], f"/home/gabriel/single_image_{j}.png")
            # save_image(single_image[0], f"/home/gabriel/single_image_{j}.png")

        # end_time=time.time()
        # time_elapsed=(end_time-start_time_total)/10
        # np.save('/home/gabriel/times_forward/1K_2_chunks.npy',time_elapsed/1000)
        return single_image,interims

    
    def forward_parallel2(self, x, iters_to_do, numchunks, overlap, device, interims=None, **kwargs):
        width = self.width
        total_overlap = (numchunks - 1) * overlap
        height = x.shape[2]
        chunk_height = (height + total_overlap) // numchunks
        gpu_count = torch.cuda.device_count()
        all_outputs = torch.zeros(x.shape[0], iters_to_do, 2, x.shape[3], x.shape[3], device=x.device)
        chunks = split_image_into_chunks(x, numchunks, overlap)
        numchunks = chunks.shape[0]
        outputs_conv= torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head3 = torch.zeros(numchunks, x.shape[0], 2, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head2 = torch.zeros(numchunks, x.shape[0], 8, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head1 = torch.zeros(numchunks, x.shape[0], 32, chunk_height, x.shape[3],device=x.device)

        start_time_total = time.time()
        self.warm_up_all_convolutions(numchunks, gpu_count, device, chunks,outputs_conv_head1,outputs_conv_head2,outputs_conv)

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')


        # torch.save(x, '/home/gabriel/tensors/x.pt')
        # are_equal = torch.equal(x_correct, x)


        
        if interims is None:
            initial_thought = self.projection(x)
            interims = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # are_equal = torch.equal(initial_thought_correct, interims)
        

        for j in range(iters_to_do):

            chunks_interims=split_image_into_chunks(interims, numchunks, overlap)
            futures = [None] * numchunks
            for i in range(numchunks):
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                chunk = chunks_interims[i].to(chunk_device)
                recall_chunk = chunks[i].to(chunk_device)
                self.to(chunk_device)
                futures[i] = torch.jit.fork(self.process_chunk1, chunk, recall_chunk)

            for i in range(numchunks):
                outputs_conv[i] = torch.jit.wait(futures[i])


            interims = combine_chunks(outputs_conv, numchunks, overlap, width,cancelation_height=5)
            # are_equal = torch.equal(interims, conv_1_2_correct)
            # if not are_equal:
            #     diff = torch.abs(interims- conv_1_2_correct)
            #     max_diff = torch.max(diff)

            chunks_interims = split_image_into_chunks(interims, numchunks, overlap)

            futures = [None] * numchunks
            for i in range(numchunks):
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                chunk = chunks_interims[i].to(chunk_device)
                self.to(chunk_device)
                futures[i] = torch.jit.fork(self.process_chunk2, chunk)

            for i in range(numchunks):
                outputs_conv_head3[i] = torch.jit.wait(futures[i])

            outputs=combine_chunks(outputs_conv_head3, numchunks, overlap, 2,cancelation_height=5)
            # are_equal = torch.equal(outputs, head3_correct)
            # if not are_equal:
            #     diff = torch.abs(outputs - head3_correct)
            #     max_diff = torch.max(diff)

            all_outputs[:,j]=outputs
            
        single_image = apply_mask(x, all_outputs, iters_to_do-1)
        end_time=time.time()
        time_elapsed=(end_time-start_time_total)/10
        # np.save('/home/gabriel/times_forward/1K_2_chunks_2_splits.npy',time_elapsed/1000)
        return single_image,interims

    def forward_parallel1(self, x, iters_to_do, numchunks, overlap, device, interims=None, **kwargs):
        width = self.width
        total_overlap = (numchunks - 1) * overlap
        height = x.shape[2]
        chunk_height = (height + total_overlap) // numchunks
        gpu_count = torch.cuda.device_count()
        all_outputs = torch.zeros(x.shape[0], iters_to_do, 2, x.shape[3], x.shape[3], device=x.device)
        chunks = split_image_into_chunks(x, numchunks, overlap)
        numchunks = chunks.shape[0]
        outputs_conv= torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head3 = torch.zeros(numchunks, x.shape[0], 2, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head2 = torch.zeros(numchunks, x.shape[0], 8, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head1 = torch.zeros(numchunks, x.shape[0], 32, chunk_height, x.shape[3],device=x.device)

        # start_time_total= time.time()
        self.warm_up_all_convolutions(numchunks, gpu_count, device, chunks,outputs_conv_head1,outputs_conv_head2,outputs_conv)

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')


        # torch.save(x, '/home/gabriel/tensors/x.pt')
        # are_equal = torch.equal(x_correct, x)

        

        start_time_total = time.time()
        if interims is None:
            initial_thought = self.projection(x)
            interims = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        

        for j in range(iters_to_do):

            chunks_interims=split_image_into_chunks(interims, numchunks, overlap)
            futures1 = [None] * numchunks
            
            for i in range(numchunks):
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                chunk = chunks_interims[i].to(chunk_device)
                recall_chunk = chunks[i].to(chunk_device)
                self.to(chunk_device)
                futures1[i] = torch.jit.fork(self.process_chunk_total, chunk, recall_chunk)

            for i in range(numchunks):
                interim_thought, out = torch.jit.wait(futures1[i])
                outputs_conv[i] = interim_thought
                outputs_conv_head3[i] = out


            interims = combine_chunks(outputs_conv, numchunks, overlap, width,cancelation_height=8)
            outputs = combine_chunks(outputs_conv_head3, numchunks, overlap, 2,cancelation_height=8)
            # are_equal = torch.equal(interims, conv_1_2_correct)

            # if not are_equal:
            #     diff = torch.abs(interims- conv_1_2_correct)
            #     max_diff = torch.max(diff)

            # are_equal = torch.equal(outputs, head3_correct)
            # if not are_equal:
            #     diff = torch.abs(outputs - head3_correct)
            #     max_diff = torch.max(diff)

            all_outputs[:,j]=outputs
            
        single_image = apply_mask(x, all_outputs, iters_to_do-1)
        end_time=time.time()
        time_elapsed=(end_time-start_time_total)/10
        # np.save('/home/gabriel/times_forward/2_chunks_1_splits.npy',time_elapsed/1000)
        return single_image,interims

    def forward_parallel1_1gpu(self, x, iters_to_do, numchunks, overlap, device, interims=None, **kwargs):
        width = self.width
        total_overlap = (numchunks - 1) * overlap
        height = x.shape[2]
        chunk_height = (height + total_overlap) // numchunks
        gpu_count = torch.cuda.device_count()
        all_outputs = torch.zeros(x.shape[0], iters_to_do, 2, x.shape[3], x.shape[3], device=x.device)
        chunks = split_image_into_chunks(x, numchunks, overlap)
        numchunks = chunks.shape[0]
        outputs_conv= torch.zeros(numchunks, x.shape[0], width, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head3 = torch.zeros(numchunks, x.shape[0], 2, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head2 = torch.zeros(numchunks, x.shape[0], 8, chunk_height, x.shape[3],device=x.device)
        outputs_conv_head1 = torch.zeros(numchunks, x.shape[0], 32, chunk_height, x.shape[3],device=x.device)

        # start_time_total= time.time()
        self.warm_up_all_convolutions(numchunks, gpu_count, device, chunks,outputs_conv_head1,outputs_conv_head2,outputs_conv)

        # x_correct = torch.load('/home/gabriel/tensors/x.pt')
        # initial_thought_correct = torch.load('/home/gabriel/tensors/projection.pt')
        # conv_1_2_correct = torch.load('/home/gabriel/tensors/conv_1_2.pt')
        # head3_correct = torch.load('/home/gabriel/tensors/head3.pt')


        # torch.save(x, '/home/gabriel/tensors/x.pt')
        # are_equal = torch.equal(x_correct, x)

        

        start_time_total = time.time()
        if interims is None:
            initial_thought = self.projection(x)
            interims = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        

        for j in range(iters_to_do):

            chunks_interims=split_image_into_chunks(interims, numchunks, overlap)
            futures1 = [None] * numchunks
            
            for i in range(numchunks):
                chunk = chunks_interims[i]
                recall_chunk = chunks[i]
                futures1[i] = torch.jit.fork(self.process_chunk_total, chunk, recall_chunk)

            for i in range(numchunks):
                interim_thought, out = torch.jit.wait(futures1[i])
                outputs_conv[i] = interim_thought
                outputs_conv_head3[i] = out


            interims = combine_chunks(outputs_conv, numchunks, overlap, width,cancelation_height=8)
            outputs = combine_chunks(outputs_conv_head3, numchunks, overlap, 2,cancelation_height=8)
            # are_equal = torch.equal(interims, conv_1_2_correct)

            # if not are_equal:
            #     diff = torch.abs(interims- conv_1_2_correct)
            #     max_diff = torch.max(diff)

            # are_equal = torch.equal(outputs, head3_correct)
            # if not are_equal:
            #     diff = torch.abs(outputs - head3_correct)
            #     max_diff = torch.max(diff)

            all_outputs[:,j]=outputs
            
        single_image = apply_mask(x, all_outputs, iters_to_do-1)
        end_time=time.time()
        time_elapsed=(end_time-start_time_total)/10
        # np.save('/home/gabriel/times_forward/1K_8_chunks_1_splits_1GPU.npy',time_elapsed/1000)
        return single_image,interims
    

    def process_chunk_total(self, interim_thought, recall_chunk):
        interim_thought = self.special_forward_conv_recall(recall_chunk, interim_thought)
        interim_thought = self.recur_block_0.forward_pieces1(interim_thought)
        interim_thought = self.recur_block_0.forward_pieces2(interim_thought)
        interim_thought = self.recur_block_1.forward_pieces1(interim_thought)
        interim_thought = self.recur_block_1.forward_pieces2(interim_thought)
        out = self.head1(interim_thought)
        out = self.head2(out)
        out = self.head_conv3(out)
        return interim_thought,out
    
    def process_chunk1(self, interim_thought, recall_chunk):
        interim_thought = self.special_forward_conv_recall(recall_chunk, interim_thought)
        interim_thought = self.recur_block_0.forward_pieces1(interim_thought)
        interim_thought = self.recur_block_0.forward_pieces2(interim_thought)
        interim_thought = self.recur_block_1.forward_pieces1(interim_thought)
        interim_thought = self.recur_block_1.forward_pieces2(interim_thought)
        return interim_thought
    
    def process_chunk2(self, interim_thought):
        out = self.head1(interim_thought)
        out = self.head2(out)
        out = self.head_conv3(out)
        return out

    def special_forward_proj(self,x):
        out = self.projection(x)
        return out

    def special_forward_recur_block_0_1(self, interim_thought):
        out = self.recur_block_0.forward_pieces1(interim_thought)
        return out
    
    def special_forward_recur_block_0_2(self, interim_thought):
        out = self.recur_block_0.forward_pieces2(interim_thought)
        return out
    
    def special_forward_recur_block_1_1(self, interim_thought):
        out = self.recur_block_1.forward_pieces1(interim_thought)
        return out
    
    def special_forward_recur_block_1_2(self, interim_thought):
        out = self.recur_block_1.forward_pieces2(interim_thought)
        return out
    
    
    def special_forward_conv_recall(self, chunks, interim_thought):
        interim_thought = torch.cat([interim_thought, chunks], 1)
        out = self.conv_recall(interim_thought)
        return out

    def special_forward_head1(self, interim_thought):
        out = self.head1(interim_thought)
        return out

    def special_forward_head2(self, interim_thought):
        out = self.head2(interim_thought)
        return out
    
    def special_forward_head3(self, interim_thought):
        out = self.head_conv3(interim_thought)
        return out

    def warm_up_all_convolutions(self,numchunks, gpu_count, device, chunks_orig, outputs_conv_head1,outputs_conv_head2,outputs_conv):
        # Define the warm-up process for each convolution layer
        def warm_up_layer(forward_function, chunks, chunks_interims=None):
            futures = [None]*numchunks
            for i in range(numchunks):
                chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
                chunk = chunks[i].to(chunk_device)
                self.to(chunk_device)
                
                if chunks_interims is not None:
                    interim = chunks_interims[i].to(chunk_device)
                    futures[i]=torch.jit.fork(forward_function, chunk, interim)
                else:
                    futures[i]=torch.jit.fork(forward_function, chunk)
                    
            for i in range(numchunks):
                torch.jit.wait(futures[i])

        # Warm-up for all layers
        warm_up_layer(self.special_forward_proj, chunks_orig)
        warm_up_layer(self.special_forward_conv_recall, outputs_conv, chunks_orig)
        warm_up_layer(self.special_forward_recur_block_0_1, outputs_conv)
        warm_up_layer(self.special_forward_recur_block_0_2, outputs_conv)
        warm_up_layer(self.special_forward_recur_block_1_1, outputs_conv)
        warm_up_layer(self.special_forward_recur_block_1_2, outputs_conv)
        warm_up_layer(self.special_forward_head1, outputs_conv)
        warm_up_layer(self.special_forward_head2, outputs_conv_head1)
        warm_up_layer(self.special_forward_head3, outputs_conv_head2)
        self.to('cuda:0')

def apply_mask(inputs, all_outputs,j):
    channel_sum = torch.sum(inputs, dim=1)
    just_path_single = (channel_sum > 0).float().unsqueeze(1).expand(-1, 3, -1, -1)
    single_predicted = all_outputs[:, j, 1]
    reshape_single = single_predicted.view(inputs.size(0), inputs.shape[2], inputs.shape[3]).unsqueeze(1).expand(-1, 3, -1, -1)
    overlayed_maze_single = reshape_single * inputs
    green_mask = (inputs[:, 1, :, :] == 1) & (inputs[:, 0, :, :] == 0) & (inputs[:, 2, :, :] == 0)
    overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
    overlayed_maze_single = overlayed_maze_single * inputs
    a_single = torch.clamp(overlayed_maze_single + just_path_single * 0.2, 0, 1)
    return a_single

def process_chunk_wrapper(queue, i, process_chunk_total, chunk, recall_chunk):
    result = process_chunk_total(chunk, recall_chunk)
    queue.put((i, result))

def combine_chunks(outputs, num_chunks, overlap, out_width,cancelation_height=1):
    # import pandas as pd
    num_chunks = outputs.shape[0]
    batch_size, _, chunk_height, width = outputs[0].shape
    stride = chunk_height - overlap

    # Create a tensor to hold the combined output
    output_height = stride * (num_chunks - 1) + chunk_height
    output_grid = torch.zeros((batch_size, out_width, output_height, width), device=outputs[0].device)      

    # with open('/home/gabriel/outputs.csv', 'w') as f:
    for i in range(num_chunks):
        chunk = outputs[i]
        # tensor_slice = chunk[0, 0].cpu().numpy()
        # df = pd.DataFrame(tensor_slice)
        # df.to_csv(f, index=False, header=f.tell()==0)  # Write header only once

        start_idx = i * stride
        # end_idx = start_idx + chunk_height

        if i == 0:
            output_grid[:, :, 0:chunk_height-cancelation_height, :] = chunk[:, :, 0:chunk_height-cancelation_height, :]

        elif i == num_chunks - 1:
            output_grid[:, :, start_idx+cancelation_height:, :] = chunk[:, :, cancelation_height:chunk_height, :]

        else:
            output_grid[:, :, start_idx+cancelation_height:start_idx+chunk_height-cancelation_height, :] = chunk[:, :, cancelation_height:chunk_height-cancelation_height, :]

            
            # tensor_slice = output_grid[0, 0].cpu().numpy()
            # df = pd.DataFrame(tensor_slice)
            # df.to_csv(f, index=False, header=f.tell()==0)  # Write header only once

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

def dt_net_2d_parallel(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)

def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)