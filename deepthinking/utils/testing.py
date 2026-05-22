""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import einops
import torch
from icecream import ic
import torch.jit
from tqdm import tqdm
from torchvision.utils import save_image
import cv2

import os
import errno
# import shutil
# import argparse
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pickle
import gc
import time
import imageio 
import imageio.v3 as iio
# Ignore statements for pylint:
#     Too many branches (R0912), Too many sinsteadtatements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114
# os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '0.7'

def test(net, loaders, mode, iters, problem, device,numchunks,overlap,terminals,parallel,termination,time_eval,acc_eval):
    accs = []
    for loader in loaders:
        if termination and parallel==False:
           accuracy = test_default_TC(net, loader, iters,device,terminals,time_eval,acc_eval) #add numchunks,overlap
            # accuracy = test_default(net, loader, iters, device,terminals,time_eval,acc_eval)
            # accuracy=test_default_and_plot(net, loader, iters, device,terminals,time_eval,acc_eval)
        elif not termination and parallel==False:
            accuracy = test_default(net, loader, iters, problem, device,terminals,time_eval,acc_eval)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)
        elif mode == "default" and parallel==True:
            accuracy = test_default_parallel(net, loader, iters, device,numchunks,overlap,terminals,time_eval,acc_eval)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs

def big(x):
    # expanded_img = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    expanded_img = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    expanded_img = expanded_img.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    # expanded_img = F.interpolate(x, scale_factor=2, mode='nearest')
    return expanded_img

def process_image(img_tensor):
    # Assuming img_tensor is 1 x h x w
    img_np = img_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension
    edges = cv2.Canny((img_np * 255).astype('uint8'), 100, 200)
    edges_tensor = torch.from_numpy(edges).float().unsqueeze(0) / 255  # Adding channel dimension back
    return edges_tensor

# save_image_self(torch.stack(saved[k]), f'../../solution_1/output_mazes_{k}.png', nrow=len(saved[k]) // 2)
def save_image_self(images, path, nrow):
    first_half, second_half = torch.split(images, split_size_or_sections=nrow, dim=0)
    second_half = torch.flip(second_half, dims=[0])
    images = torch.cat([first_half, second_half], dim=0)
    save_image(images, path, nrow=nrow)

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted

def test_default_dynamic(net, testloader, iters, device,terminals,time_eval,acc_eval):

    if acc_eval:
        max_iters = 70
    elif time_eval:
        max_iters = 50
    else:
        max_iters = max(iters)

    net.eval()
    initial_iters=20
    # corrects = torch.zeros(max_iters/2)
    corrects_dyn=torch.zeros(max_iters)
    durations=[]
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            start_time=time.time()
            inputs1=inputs[0].unsqueeze(0)
            inputs2=inputs[1].unsqueeze(0)
            all_outputs11, interims_1= net(inputs1, iters_to_do=initial_iters)
            all_outputs21, interims_2= net(inputs2, iters_to_do=initial_iters)
            all_outputs12,_= net(inputs2, iters_to_do=max_iters,interim_thought=interims_1)
            all_outputs22,_= net(inputs1, iters_to_do=max_iters,interim_thought=interims_2)
            all_outputs1=torch.cat((all_outputs11,all_outputs12),dim=1)
            all_outputs2=torch.cat((all_outputs21,all_outputs22),dim=1)
            all_outputs=torch.cat((all_outputs1,all_outputs2),dim=0)
            end_time=time.time()
            duration=((end_time-start_time)/all_outputs.shape[0])/max_iters
            durations.append(duration)

            creategifs(all_outputs11,all_outputs21,all_outputs12,all_outputs22,inputs)

            for i in range(all_outputs11.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, "mazes")
                switched_targets = targets[[1, 0], :]
                targets = switched_targets.view(targets.size(0), -1)
                corrects_dyn[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)
            
    
    accuracy = 100.0 * corrects_dyn / total
    ret_acc = {}

    ####################################################################
    if time_eval:
        np.save(f'/home/gabriel/durations_net/no_TC_{terminals}_G_21_{max_iters}_iters.npy',durations)
    
    if acc_eval:
        np.save(f'/home/gabriel/convergence_net/no_termination_condition/{terminals}G.npy',np.array(accuracy))
    ######################################################################
    
    for ite in iters:
        ret_acc[ite] =  accuracy[ite-1].item()
    return ret_acc

def creategifs(all_outputs11,all_outputs21,all_outputs12,all_outputs22,inputs):

    first_tensor=torch.cat((all_outputs11,all_outputs12),dim=1).squeeze(0)[:,1]
    second_tensor=torch.cat((all_outputs21,all_outputs22),dim=1).squeeze(0)[:,1]

    channel_sum = torch.sum(inputs, dim=1)
    just_path = (channel_sum > 0).float()
    

    just_path = just_path.unsqueeze(1).expand(-1, 3, -1, -1)

    modified_tensor1 = first_tensor.unsqueeze(1).expand(-1, 3, -1, -1).clone()
    modified_tensor2 = second_tensor.unsqueeze(1).expand(-1, 3, -1, -1).clone()

    green_mask1 = (inputs[0, 1, :, :] == 1) & (inputs[0, 0, :, :] == 0) & (inputs[0, 2, :, :] == 0)
    green_mask2 = (inputs[1, 1, :, :] == 1) & (inputs[1, 0, :, :] == 0) & (inputs[1, 2, :, :] == 0)

    black_mask1 = (inputs[0, 0, :, :] == 0) & (inputs[0, 1, :, :] == 0) & (inputs[0, 2, :, :] == 0)  
    black_mask2 = (inputs[1, 0, :, :] == 0) & (inputs[1, 1, :, :] == 0) & (inputs[1, 2, :, :] == 0)

    modified_tensor1[0:20, 1, :, :].masked_fill_(green_mask1, 1)
    modified_tensor1[0:20, 0, :, :].masked_fill_(green_mask1, 0)
    modified_tensor1[0:20, 2, :, :].masked_fill_(green_mask1, 0)

    modified_tensor1[0:20, :, :, :].masked_fill_(black_mask1, 0)

    modified_tensor1[20:70, 1, :, :].masked_fill_(green_mask2, 1)
    modified_tensor1[20:70, 0, :, :].masked_fill_(green_mask2, 0)
    modified_tensor1[20:70, 2, :, :].masked_fill_(green_mask2, 0)

    modified_tensor1[20:70, :, :, :].masked_fill_(black_mask2, 0)

    modified_tensor2[0:20, 1, :, :].masked_fill_(green_mask2, 1)
    modified_tensor2[0:20, 0, :, :].masked_fill_(green_mask2, 0)
    modified_tensor2[0:20, 2, :, :].masked_fill_(green_mask2, 0)

    modified_tensor2[0:20, :, :, :].masked_fill_(black_mask2, 0)

    modified_tensor2[20:70, 1, :, :].masked_fill_(green_mask1, 1)
    modified_tensor2[20:70, 0, :, :].masked_fill_(green_mask1, 0)
    modified_tensor2[20:70, 2, :, :].masked_fill_(green_mask1, 0)

    modified_tensor2[20:70, :, :, :].masked_fill_(black_mask1, 0)

    modified_tensor1[:] = torch.clamp(modified_tensor1[:], 0, 1)
    modified_tensor2[:] = torch.clamp(modified_tensor2[:], 0, 1)

   
    frames1 = []
    frames2 = []
    for i in range(modified_tensor1.size(0)):
        # frame = modified_tensor1[i] + just_path[0] * 0.2  # Add `just_path` to all frames
        # frame = torch.clamp(frame, 0, 1)  # Ensure values are within valid range
        if i<20:
            frame1=modified_tensor1[i] + just_path[0] * 0.2
            frame1 = torch.clamp(frame1, 0, 1)
            frame2 = modified_tensor2[i]+just_path[1]*0.2  # Add `just_path` to all frames
            frame2 = torch.clamp(frame2, 0, 1)  # Ensure values are within valid range
        else:
            frame1=modified_tensor1[i] + just_path[1] * 0.2
            frame1 = torch.clamp(frame1, 0, 1)
            frame2 = modified_tensor2[i]+just_path[0]*0.2
            frame2 = torch.clamp(frame2, 0, 1)
    
        frames1.append((frame1.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))  # Convert to numpy
        frames2.append((frame2.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))  # Convert to numpy

    gif_path = '/home/gabriel/Lydia/animated_tensor.gif'
    iio.imwrite(gif_path, frames1, format="GIF", duration=100)
    gif_path = '/home/gabriel/Lydia/animated_tensor2.gif'
    iio.imwrite(gif_path, frames2, format="GIF", duration=100)

    return

def test_default(net, testloader, iters, problem, device,terminals,time_eval,acc_eval):

    if acc_eval:
        max_iters = 70
    elif time_eval:
        max_iters = 50
    else:
        max_iters = max(iters)

    net.eval()
    corrects = torch.zeros(max_iters)
    durations=[]
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            start_time=time.time()
            all_outputs,_ = net(inputs, iters_to_do=max_iters)
            end_time=time.time()
            duration=((end_time-start_time)/all_outputs.shape[0])/max_iters
            durations.append(duration)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)
            
    
    accuracy = 100.0 * corrects / total
    ret_acc = {}

    ####################################################################
    if time_eval:
        np.save(f'/home/gabriel/durations_net/no_TC_{terminals}_G_21_{max_iters}_iters.npy',durations)
    
    if acc_eval:
        np.save(f'/home/gabriel/convergence_net/no_termination_condition/{terminals}G.npy',np.array(accuracy))
    ######################################################################
    
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc

def test_default_TC(net, testloader, iters, device, terminals,time_eval,acc_eval):
    if acc_eval:
        max_iters = 20
    elif time_eval:
        max_iters = 4
    else:
        max_iters = int(max(iters)/10)
        # max_iters=20

    net.eval()
    total = 0
    corrects_per_iteration = torch.zeros(max_iters*10)
    # convergence_per_iteration = torch.zeros(max_iters*10)
    # corrects = torch.zeros(max_iters)
    j=0
    mistakes=[]
    durations=[]

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            j+=1
            # save_image(inputs[0], f'/home/gabriel/input_{j}.png')
            # save_image(targets.float(), f'/home/gabriel/target.png')
            corrects = torch.zeros(targets.size(0), max_iters*10)
            convergences = torch.zeros(targets.size(0), max_iters*10)
            predicted_targets = torch.ones(targets.size(), device=device)
            interims=None
            first=1

            # all_outputs,interims= net(inputs,max_iters, interims)

            
            for i in range(max_iters+1):
                start_time_total=time.time()
                all_outputs,interims = net(inputs,10+(first*20),interims)
                # save_image(all_outputs[0, -1, 1], f'/home/gabriel/trial_single.png')
                single_image=apply_mask(inputs,all_outputs)
                # save_image(single_image[0], f'/home/gabriel/trial_single_masked.png')
                # end_time=time.time()
                # time_elapsed=(end_time-start_time)/10
                # np.save('/home/gabriel/times_forward/1k_no_parallel.npy',time_elapsed/1000)
                single_target, finishidx, visited_positions = check_termination_multiple(single_image)
                # save_image(single_target, f'/home/gabriel/trial_single_afterTC.png')
                predicted_targets[:, :] = single_target.squeeze()
                first=0

            # save_image(all_outputs[0, -1, 1], '/home/gabriel/trial_single.png')
            # duration=((end_time-start_time)/all_outputs.shape[0])/max_iters
            # durations.append(duration)

            # for i in range(all_outputs.size(1)):
            #     outputs = all_outputs[:, i]
            #     predicted = get_predicted(inputs, outputs, problem)
            #     targets = targets.view(targets.size(0), -1)
            #     corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            # total += targets.size(0)
            
    # #pdb.set_trace()######################################################
    # #np.save('/home/gabriel/durations_net/5_G_21_1gpu.npy',durations)
    # ######################################################################
    # accuracy = 100.0 * corrects / total
    # ret_acc = {}
    # for ite in iters:
    #     ret_acc[ite] = accuracy[ite-1].item()
    # return ret_acc

                if i==max_iters and finishidx==False:
                    duration_total=time.time()-start_time_total
                    durations.append(duration_total)
                    # save_image(single_image[0], f'/home/gabriel/trial.png')
                    # save_image(targets[0].float(), f'/home/gabriel/target.png')
                    # save_image(inputs[0], f'/home/gabriel/input.png')
                    mistakes.append((j,inputs.cpu().numpy()[0],targets.cpu().numpy()[0],single_image.cpu().numpy()[0]))
                    break
                if finishidx:
                    duration_total=time.time()-start_time_total
                    durations.append(duration_total)
                    num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                    num_ones = targets.sum(dim=[1, 2]).to(device)
                    correct = int(num_ones_predicted <= (num_ones + 8))
                    convergences[:, i:max_iters*10] = 1
                    if correct == 1:
                        corrects[:, i:max_iters*10] = 1
                        break
                    else:
                        if i==max_iters:
                            mistakes.append((j,inputs.cpu().numpy()[0],targets.cpu().numpy()[0],single_image.cpu().numpy()[0]))
                            hello=1
                        # save_image(single_image[0], f'/home/gabriel/trial.png')
                        # save_image(inputs[0], f'/home/gabriel/input.png')
                        # save_image(targets[0].float(), f'/home/gabriel/target.png')
                        # save_image(single_target, f'/home/gabriel/trial_target.png')
                        # save_image(targets[0].float(), f'/home/gabriel/target.png')
                        # mistakes.append((inputs.cpu().numpy()[0],targets.cpu().numpy()[0]))
                        # break

            for i in range(max_iters*10):
                corrects_per_iteration[i] += corrects[:, i].sum().item()
                # convergence_per_iteration[i] += convergences[:, i].sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects_per_iteration / total
    # save_image(predicted_targets[0], f'/home/gabriel/predicted_target.png')
        # convergence=100.0 * convergence_per_iteration / total
        # accuracy = 100.0 * corrects / total
        # np.save('/home/gabriel/convergence_net/no_termination_condition/7G.npy',np.array(accuracy))
    # print(accuracy)
    # print(len(mistakes))####################################################################
        # print(convergence)
        # print(len(mistakes),accuracy[-1])
    np.save(f'/home/gabriel/mistakes_retrain/zind/big_floorplans/{terminals}G.npy',np.array(mistakes))#######################
    
    ####################################################################
    if time_eval:
        np.save(f'/home/gabriel/durations_net/TC_{terminals}_G_21_{max_iters*10}_iters.npy',durations)
    
    if acc_eval:
        # np.save(f'/home/gabriel/convergence_net/termination_condition/{terminals}G.npy',np.array(accuracy))
        hello=1
    ######################################################################

    ret_acc = {}
    # print(accuracy)
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    ret_acc[-1]=accuracy[-1].item()
    return ret_acc

def test_default_parallel(net, testloader, iters, device,numchunks,overlap,terminals,time_eval,acc_eval):
    max_iters = max(iters)
    net.eval()
    corrects_per_iteration = torch.zeros(max_iters)
    iters_to_do=10
    # gpu_count = torch.cuda.device_count()
    total=0
    durations=[]
    # warm_up_all_convolutions(numchunks, gpu_count, device,net,overlap)

    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            corrects = torch.zeros(targets.size(0), max_iters)
            predicted_targets = torch.ones(targets.size(), device=device)
            interims=None

            # chunks=split_image_into_chunks(inputs, numchunks, overlap)


           
            

            for i in range(iters_to_do, max_iters + 1, iters_to_do):

                # outputs, interims= forward_parallel(
                #     net, chunks, device, gpu_count, iters_to_do, interims
                # )

                start_time_total = time.time()
                single_image, interims= net.forward_parallel1_1gpu(inputs, iters_to_do, numchunks, overlap,device, interims)
                # end_time=time.time()
                # time_elapsed=(end_time-start_time_total)/10
                # np.save('/home/gabriel/times_forward/1K_2_chunks_2_splits.npy',time_elapsed/1000)
                # save_image(single_image, f'/home/gabriel/trial_single.png')

                # all_outputs = combine_chunks(outputs, numchunks, overlap)

                # single_image = apply_mask(inputs, all_outputs)
                

                single_target, finishidx, visited_positions = check_termination_multiple(single_image)
                predicted_targets[:, :] = single_target.squeeze()
                

                if i==max_iters and finishidx==False:
                    duration_total=time.time()-start_time_total
                    durations.append(duration_total)
                    break
                if finishidx:
                    duration_total=time.time()-start_time_total
                    durations.append(duration_total)
                    num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                    num_ones = targets.sum(dim=[1, 2]).to(device)
                    correct = int(num_ones_predicted <= (num_ones + 8))
                    if correct == 1:
                        corrects[:, i:max_iters] = 1
                    break

            for i in range(max_iters):
                corrects_per_iteration[i] += corrects[:, i].sum().item()

            total += targets.size(0)

    # print(f"Total time: {np.mean(duration_total):.4f} seconds")
    accuracy = 100.0 * corrects_per_iteration / total
    # np.save('/home/gabriel/mistakes_retrain/4G.npy',np.array(mistakes))
    # durations_mean=np.mean(np.array(durations))
    # np.save('/home/gabriel/durations_net_corrected/4G_10skip.npy',np.array(durations_mean))
    # print(accuracy)

    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc

def test_default_with_termination_parallel(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    total = 0
    end=50
    corrects = torch.zeros(end,device=device)
    

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):

            inputs, targets = inputs.to(device), targets.to(device)
            predicted_targets=torch.ones((inputs.size(0), inputs.size(2), inputs.size(3)),device=device)
            interim=None
            finish = [False] * inputs.size(0)
            iteration_count = 0
            corrects_batch = torch.zeros(inputs.size(0),end,device=device)
            remaining_indices = list(range(inputs.size(0)))
            
            start_time=time.time()
            while remaining_indices and iteration_count < 10:
                

                # current_inputs = inputs[remaining_indices]
                # if interim is not None:
                #     interim = interim[remaining_indices]
                
                all_outputs, interim = net(inputs, iters_to_do=max_iters,interim_thought=interim) #add predicted targets
                
                #print(duration)
                # for i in range(all_outputs.size(1)):
                #     outputs = all_outputs[:, i]
                #     predicted = get_predicted(inputs, outputs, problem)
                #     targets = targets.view(targets.size(0), -1)
                #     corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()
                all_outputs1=F.softmax(all_outputs, dim=2)
                channel_sum = torch.sum(inputs, dim=1)
                just_path_single = (channel_sum > 0).float()
                just_path_single = just_path_single.unsqueeze(1).expand(-1, 3, -1, -1)
                single_predicted=all_outputs1[:,-1,-1]
                reshape_single = single_predicted.view(all_outputs1.size(0), inputs.shape[2], inputs.shape[3])
                reshape_single = reshape_single.unsqueeze(1).expand(-1, 3, -1, -1)
                overlayed_maze_single = reshape_single * inputs
                green_mask = (inputs[:, 1, :, :] == 1) & (inputs[:, 0, :, :] == 0) & (inputs[:, 2, :, :] == 0)
                overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
                overlayed_maze_single = overlayed_maze_single * inputs
                a_single = overlayed_maze_single + just_path_single * 0.2
                a_single = torch.clamp(a_single, 0, 1)
                finished_indices=[]
                for i in remaining_indices:
                    visited_positions=set()
                    single_image = a_single[i].unsqueeze(0)
                    single_target, finishidx, visited_positions = check_termination_multiple(single_image)
                    # save_image(single_target, f'/home/gd27/trial_single_target.png')
                    # save_image(single_image, f'/home/gd27/trial_single.png')
                    # num_ones_predicted = single_target.sum(dim=[1, 2])
                    # num_ones = targets[i].sum(dim=[1, 2])
                    # correct = num_ones_predicted <= (num_ones + 4)
                    # corrects[i] = correct
                    if finishidx:
                        predicted_targets[i,:,:]=single_target.squeeze()
                        num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                        num_ones = targets[i].sum(dim=[0, 1]).to(device)
                        correct = int(num_ones_predicted <= (num_ones + 4))
                        corrects_batch[i,(iteration_count+1)*max_iters:end] = correct
                        finished_indices.append(i)

                remaining_indices = [idx for idx in remaining_indices if idx not in finished_indices]
                iteration_count += 1
            
            end_time=time.time()
            duration=((end_time-start_time)/corrects_batch.size(0))
            durations.append(duration)
            print(duration)

            corrects+=corrects_batch.sum(dim=0)
            total += targets.size(0)
    
    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc

    #         for i in range(predicted_targets.size(0)):
    #             predicted = predicted_targets[:, i].to(device) # Predictions for iteration i
    #             # correct = (predicted == targets_expanded[:, i]).all(dim=2).all(dim=1)  # Check if all elements match
    #             # corrects[i] += correct.sum().item()  # Count correct predictions

    #             num_ones_predicted = predicted.sum(dim=[1, 2])
    #             num_ones_target = targets_expanded[:, i].sum(dim=[1, 2])

    #             # Check if the number of ones in predicted is less than or equal to the number of ones in target plus four
    #             correct = num_ones_predicted <= (num_ones_target + 4)
    #             corrects[i] += correct.sum().item() 

    #         # for j in range(predicted_targets.size(0)):
    #         #     save_image(predicted_targets[j,99,:,:], f'/home/gd27/mazes/temp2/predictions/prediction_{j}.png')

    #         #print(corrects)

    #         total += targets.size(0)
            
    # #pdb.set_trace()######################################################
    # #np.save('/home/gd27/durations_net/8_G_21_50_iters_128_5skip_fast.npy',durations)
    # ######################################################################
    

    # accuracy = 100.0 * corrects / total
    # ret_acc = {}
    # for ite in iters:
    #     ret_acc[ite] = accuracy[ite-1].item()
    # return ret_acc


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}

    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc

def test_default_with_termination(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0
    results=[]
    j=0
    wrong_inputs=[]
    wrong_targets=[]
    right_inputs=[]

    dir_name = "Mazes_stop"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            ########################################################## 
            for k in range(inputs.size(0)):
                inputs_single=inputs[k].unsqueeze(0).to(device)
                channel_sum = torch.sum(inputs_single, dim=1)
                just_path_single = (channel_sum > 0).float()
                just_path_single = just_path_single.unsqueeze(1).expand(-1, 3, -1, -1)
                start_time=time.time()

                for i in range(max_iters):
                    single_output=net(inputs_single, iters_to_do=i+1)
                    start_time_partial_1=time.time()  
                    single_output=single_output[:,i]
                    single_output=F.softmax(single_output, dim=1)
                    single_predicted=single_output[:,1]
                    reshape_single = single_predicted.view(single_predicted.size(0), inputs.shape[2], inputs.shape[3])
                    reshape_single = reshape_single.unsqueeze(1).expand(-1, 3, -1, -1)
                    overlayed_maze_single = reshape_single * inputs_single
                    green_mask = (inputs_single[:, 1, :, :] == 1) & (inputs_single[:, 0, :, :] == 0) & (inputs_single[:, 2, :, :] == 0)
                    overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
                    overlayed_maze_single = overlayed_maze_single * inputs_single
                    a_single = overlayed_maze_single + just_path_single * 0.2
                    a_single = torch.clamp(a_single, 0, 1)
                    #save_image(a_single, 'trial_single.png') 
                    end_time_partial_1=time.time()  
                    start_time_partial_2=time.time()     
                    single_target,finish=check_termination_multiple(a_single)
                    end_time_partial_2=time.time() 
                    duration_partial_1=(end_time_partial_1-start_time_partial_1)
                    duration_partial_2=(end_time_partial_2-start_time_partial_2)
                    if i==0:
                        end_time=time.time()
                        duration=(end_time-start_time)*(i+1)
                        # print(duration,duration_partial_1,duration_partial_2, duration_partial_1/duration*100, duration_partial_2/duration*100)
                    
                    if finish or i==max_iters-1:
                        duration=(end_time-start_time)*(i+1)
                        # print(duration,duration_partial, duration_partial/duration*100)
                        durations.append(duration)
                        reshape_single = single_target.view(single_predicted.size(0), inputs.shape[2], inputs.shape[3])
                        reshape_single = reshape_single.unsqueeze(1).expand(-1, 3, -1, -1)
                        overlayed_maze_single = (reshape_single.to(device) * inputs_single)
                        green_mask = (inputs_single[:, 1, :, :] == 1) & (inputs_single[:, 0, :, :] == 0) & (inputs_single[:, 2, :, :] == 0)
                        overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
                        overlayed_maze_single = overlayed_maze_single * inputs_single
                        final_single = overlayed_maze_single + just_path_single * 0.2
                        final_single = torch.clamp(final_single, 0, 1)
                        results.append((i+1,final_single))
                        file_name = f'batch_{j}_Maze_{k}_iteration_{i+1}.png'
                        file_path = os.path.join(dir_name, file_name)
                        # wrong_inputs.append(inputs_single[0])########################################
                        # wrong_targets.append(targets[k].unsqueeze(0).to(device))############################
                        corrects[i]+=1
                        if i==max_iters-1 and finish==False:
                            save_image(final_single, file_path)
                            save_image(a_single,os.path.join(dir_name,'trial.png'))
                            corrects[i]-=1
                            wrong_inputs.append(inputs_single[0])
                            wrong_targets.append(targets[k].unsqueeze(0).to(device))
                        break
            #     if k==20:
            #         break
            # if j==0:
            #     break
            j+=1
        
 # Clean up
    del inputs_single, single_output, reshape_single, overlayed_maze_single, green_mask, a_single, final_single
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    
    if len(wrong_inputs)>0:
        wrong_inputs_tensor = torch.stack(wrong_inputs) # Ensure tensors are on the right device
        wrong_targets_tensor = torch.stack(wrong_targets)

        num_splits = 4  # Adjust this based on your GPU's capacity
        inputs_splits = torch.chunk(wrong_inputs_tensor, num_splits, dim=0)

        all_outputs = []
        for inputs_part in inputs_splits:
            # Process each part on the GPU
            outputs_part = net(inputs_part, iters_to_do=max_iters)
            all_outputs.append(outputs_part.detach())  # Detach to avoid building up the computation graph

            # Free up memory
            del inputs_part, outputs_part
            torch.cuda.empty_cache()

        # Concatenate the results on GPU
        allwrong_outputs = torch.cat(all_outputs, dim=0).to('cuda:0')

        # Clear the list to free memory
        del all_outputs
        torch.cuda.empty_cache()

        # torch.save(allwrong_outputs, 'all_outputs.pt')
        # torch.save(wrong_inputs_tensor, 'wrong_inputs.pt')
        # torch.save(wrong_targets_tensor, 'wrong_targets.pt')
            
    # inputs = torch.load('/home/gabriel/outputs/Wrong_2G_1M/wrong_inputs.pt', map_location='cpu')
    # targets = torch.load('/home/gabriel/outputs/Wrong_2G_1M/wrong_targets.pt', map_location='cpu')
    # all_outputs=net(inputs, iters_to_do=max_iters)
    
    # torch.save(all_outputs, '/home/gabriel/outputs/Right_2G_1M/Perfect_one/Difficults/all_outputs.pt')
    # torch.save(inputs, '/home/gabriel/outputs/Right_2G_1M/Perfect_one/Difficults/right_inputs.pt')
    # torch.save(targets, '/home/gabriel/outputs/Right_2G_1M/Perfect_one/Difficults/targets.pt')

    results_file_path = os.path.join(dir_name, "results.pkl")
    with open(results_file_path, 'wb') as file:
        pickle.dump(results, file)
    
    
    cumulative_corrects = torch.cumsum(corrects, dim=0)
    ret_acc_tensor = 100 * cumulative_corrects.float() / total
    ret_acc = ret_acc_tensor.tolist()
    return ret_acc
    ############################################################### 
    

def test_default_and_plot(net, testloader, iters, device,terminals,time_eval,acc_eval):
    max_iters = 200 #max(iters)
    net.eval()
    j=0
    corrects = torch.zeros(max_iters)
    total = 0
    folder = f"solution_mi_30_ec"
    create_folder(f"../../{folder}")
    # random int to create a new folder in it
    folder = f"{folder}/{np.random.randint(1000)}"
    # create_folder(f"../../{folder}")
    # iters = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
    iters = list(range(1, max_iters + 1))

    mistakes=np.load(f'/home/gabriel/mistakes_retrain/zind/big_floorplans/{terminals}G.npy',allow_pickle=True)
    gif_indices=[]

    for i in range(len(mistakes)):
        index,_,_,_=mistakes[i]
        gif_indices.append(index)




    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            j+=1
            ########################################################## 
            # inputs_single=inputs[0].unsqueeze(0).to(device)
            # bg_inputs_single = big(inputs_single)
            # channel_sum = torch.sum(bg_inputs_single, dim=1)
            # just_path_single = (channel_sum > 0).float()
            # just_path_single = just_path_single.unsqueeze(1).expand(-1, 3, -1, -1)
            

            # for i in range(max_iters):
            #     single_output,_=net(inputs_single, iters_to_do=i+1)
            #     single_output=single_output[:,i]
            #     single_output=F.softmax(single_output, dim=1)
            #     single_predicted=single_output[:,1]
            #     reshape_single = single_predicted.view(single_predicted.size(0), inputs.shape[2], inputs.shape[3])
            #     reshape_single = reshape_single.unsqueeze(1).expand(-1, 3, -1, -1)
            #     overlayed_maze_single = reshape_single * inputs_single
            #     green_mask = (inputs_single[:, 1, :, :] == 1) & (inputs_single[:, 0, :, :] == 0) & (inputs_single[:, 2, :, :] == 0)
            #     overlayed_maze_single[:, 1, :, :].masked_fill_(green_mask, 1)
            #     overlayed_maze_single = overlayed_maze_single * inputs_single
            #     a_single = big(overlayed_maze_single) + just_path_single * 0.2
            #     a_single = torch.clamp(a_single, 0, 1)
                # save_image(a_single, 'trial_single.png')               
                # finish=check_termination(a_single)
                # if finish:

                    # break



            ########################################################### 
            all_outputs,_ = net(inputs, iters_to_do=max_iters)
            # print(inputs.shape)

            single_image=apply_mask(inputs,all_outputs)
            single_target, _,_ = check_termination_multiple(single_image)

            bg_inputs = big(inputs)
            # print(bg_inputs.shape)
            # save_image(bg_inputs, f'../../{folder}/inputs.png')
            channel_sum = torch.sum(bg_inputs, dim=1)
            just_path = (channel_sum > 0).float()
            just_path = just_path.unsqueeze(1).expand(-1, 3, -1, -1)
            # save_image(just_path, f'../../{folder}/just_path_solid.png')

            if j in gif_indices:
                saved = [[] for _ in range(inputs.size(0))]
                for i in range(inputs.size(0)):
                    saved[i].append(bg_inputs[i])

                for i in range(all_outputs.size(1)):
                    outputs = all_outputs[:, i]
                    # apply softmax to get probabilities
                    outputs = F.softmax(outputs, dim=1)
                    predicted = outputs[:, 1]
                    reshape = predicted.view(predicted.size(0), inputs.shape[2], inputs.shape[3])
                    reshape = reshape.unsqueeze(1).expand(-1, 3, -1, -1)
                    overlayed_maze = reshape * inputs

                    green_mask = (inputs[:, 1, :, :] == 1) & (inputs[:, 0, :, :] == 0) & (inputs[:, 2, :, :] == 0)
                    red_mask = (inputs[:, 0, :, :] == 1) & (inputs[:, 1, :, :] == 0) & (inputs[:, 2, :, :] == 0)

                    overlayed_maze[:, 1, :, :].masked_fill_(green_mask, 1)
                    overlayed_maze[:, 0, :, :].masked_fill_(red_mask, 1)

                    overlayed_maze = overlayed_maze * inputs

                    if i in iters:
                        for k in range(inputs.size(0)):
                            a = big(overlayed_maze[k]) + just_path[k] * 0.2
                            a = torch.clamp(a, 0, 1)
                            saved[k].append(a)
                    
                images = []
                for image_tensor in saved[k]:
                    image = image_tensor.permute(1, 2, 0).cpu().numpy()
                    images.append((image * 255).astype(np.uint8))  # Convert tensor to uint8 for GIF

                gif_path = f'/home/gabriel/mistakes_retrain/zind/big_floorplans/GIFS/{terminals}/GIF_{j}.gif'
                imageio.mimsave(gif_path, images, duration=200,loop=0) 
            


            total += targets.size(0)
            
            # break

    # for k in range(inputs.size(0)):
    #     save_image(torch.stack(saved[k]), f'../../{folder}/output_mazes_{k}.png', nrow=len(saved[k]))
        

    # for k in range(inputs.size(0)):
    #     num_images = len(saved[k])  
    #     fig, axs = plt.subplots(1, num_images, figsize=(num_images * 5, 5.8))  # Adjust figsize as needed

       
    #     for i, image_tensor in enumerate(saved[k]):
    #         image = image_tensor.permute(1, 2, 0).cpu().numpy()  
    #         axs[i].imshow(image)
            
    #         if i == 0:
    #             axs[i].set_title("Input",fontsize=30)  # First image titled "Input"
    #         else:
    #             axs[i].set_title(f'Iteration {iters[i-1]}',fontsize=30) 

    #         axs[i].axis('off')  

    #     plt.tight_layout()
    #     plt.savefig(f'../../{folder}/output_mazes_{k}.png')  
    #     plt.close(fig) 


#result for termination_condition
    # for k in range(inputs.size(0)):
    #     black_positions = (single_target[k][0] == 0)
    #     single_image[k,:, black_positions] = 0
    #     a = big(single_image[k])
    #     # a = torch.clamp(a, 0, 1)
    #     for i in range(15):
    #         saved[k].append(a)
    #     # saved[k].append(a)
        

    exit()
    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


# def create_GIF(all_outputs,single_image,single_target):

def forward_parallel(net, chunks, device, gpu_count, iters_to_do, interims):

    width=128
    outputs = torch.zeros(chunks.shape[0], chunks.shape[1],iters_to_do,2, chunks.shape[3], chunks.shape[4],device='cuda:0')
    if interims==None:
        interims=torch.zeros(chunks.shape[0], chunks.shape[1],width,chunks.shape[3], chunks.shape[4],device='cuda:0')
    futures = []
    chunk_count = chunks.shape[0]

    for i in range(chunk_count):

        chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
        chunk = chunks[i].to(chunk_device)
        interim = interims[i].to(chunk_device)
        net = net.to(chunk_device)

        futures.append(
            torch.jit.fork(net.forward, chunk, iters_to_do, interim)
        )

    for i,future in enumerate (futures):
        output, interim = torch.jit.wait(future)
        outputs[i] = output
        interims[i] = interim


    return outputs,interims

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


def Fill_target_multiple(single_output, visited_positions):
    visited_positions = torch.tensor(visited_positions)#hereeeeee
    target_tensor = torch.zeros((1, 1, single_output.shape[2], single_output.shape[3]))#hereeee

    # Generate additional positions to mark based on the visited ones
    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)#hereeee
    all_positions = visited_positions[:, None, :] + offsets[None, :, :]
    all_positions = all_positions.reshape(-1, 2)  # Flatten the positions

    # Target tensor initialization
    target_tensor = torch.zeros((1, 1, single_output.shape[2], single_output.shape[3]))#hereeee

    # Use advanced indexing to fill the target tensor
    y, x = all_positions[:, 0], all_positions[:, 1]
    target_tensor[0, 0, y, x] = 1

    return target_tensor

def check_termination_multiple(single_output):
    
    green_channel = single_output[:, 1, :, :]
    red_channel = single_output[:, 0, :, :]
    blue_channel = single_output[:, 2, :, :]
    green_threshold = 0.8
    RB_threshold = 0.5
    is_green = (green_channel > green_threshold) & (red_channel < RB_threshold) & (blue_channel < RB_threshold)
    green_pixel_positions = torch.nonzero(is_green)
    
   
    top_left_corners = identify_top_left_corners(green_pixel_positions) 
    
    #Run the check if it is over
    visited_positions,check = move_towards_whiteness_check_green_multiple(single_output, top_left_corners)
    
    single_target=Fill_target_multiple(single_output, visited_positions)
   
    
    # print(d1,d2,d3,d4,d1/dt,d2/dt,d3/dt,d4/dt)
    #save_image(single_target, '/home/gd27/trial_single_target.png')
    
    return single_target, check, visited_positions

def identify_top_left_corners(green_pixel_positions):
    green_pixel_positions = green_pixel_positions.clone().detach()
    yx_positions = green_pixel_positions[:, 1:]

    # Assuming the maximum size from green_pixel_positions to define the size of the grid
    max_y, max_x = yx_positions.max(0)[0]
    grid = torch.zeros((max_y + 2, max_x + 2), dtype=torch.bool)  # +2 to accommodate boundary conditions hereeeeee
    grid[yx_positions[:, 0], yx_positions[:, 1]] = True

    # Initialize an empty tensor for top-left corners
    top_left_corners = []

    # Vectorized check for the presence of all four corners of a 2x2 square
    for y, x in yx_positions:
        if grid[y, x]:
            # Check if the current position and the three other corners form a 2x2 square
            is_top_left = grid[y, x] & grid[y, x + 1] & grid[y + 1, x] & grid[y + 1, x + 1]
            if is_top_left:
                top_left_corners.append([y.item(), x.item()])
                # Mark the current square as processed to avoid duplicates
                grid[y:y + 2, x:x + 2] = False

    return torch.tensor(top_left_corners, device=green_pixel_positions.device)

    # Assuming image_tensor is [Batch, Channels, Height, Width]
    # Calculate the average across channels to simplify, assuming whiteness is an average of RGB values
    avg_tensor = image_tensor.mean(dim=1, keepdim=True)
    
    # Use unfold to efficiently calculate the mean for each 2x2 block
    # stride and kernel_size set to 2 to compute for non-overlapping 2x2 blocks
    whiteness_scores = avg_tensor.unfold(2, 2, 2).unfold(3, 2, 2).mean(dim=[2, 3], keepdim=False)
    
    # Normalize whiteness_scores if necessary, based on your definition of "whiteness"
    return whiteness_scores


def move_towards_whiteness_check_green_multiple(image_tensor, top_left_corners):
    directions = {'east': (0, 2), 'south': (2, 0), 'north': (-2, 0), 'west': (0, -2)}
    opposite_directions = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}
    start_position_tensor = top_left_corners[0]
    start_position = (start_position_tensor[0].item(), start_position_tensor[1].item())
    current_position = torch.tensor([start_position[0], start_position[1]])
    visited_positions = set()
    green_visited = set()
    junctions = []  # List of tuples containing position and unexplored directions
   

    exploration_visited_positions=set()
    from_junction=False
    movement=[]
    solved,remaining_corners=explore(current_position, None, directions, opposite_directions, visited_positions, green_visited, junctions, image_tensor, top_left_corners,top_left_corners.tolist(),exploration_visited_positions,from_junction,movement)

    # Backtrack from junctions if needed
    while junctions and len(remaining_corners)>0:
        position, unexplored_directions = junctions.pop()  # Use a different variable name here
        for direction_name,_ in unexplored_directions[1:]:  # Adjust variable name accordingly
            # Now, correctly use the 'directions' dictionary with the direction name
            new_dy, new_dx = directions[direction_name]
            new_position = position + torch.tensor([2*new_dy, 2*new_dx])######################here
            intermediate_position = position + torch.tensor([new_dy, new_dx])#################here
            if tuple(new_position.tolist()) not in visited_positions:
                movement=[]
                movement.append(tuple(intermediate_position.tolist()))
                from_junction=True
                solved,remaining_corners=explore(new_position, direction_name, directions, opposite_directions, visited_positions, green_visited, junctions, image_tensor, top_left_corners,remaining_corners,exploration_visited_positions,from_junction,movement)

    return list(visited_positions), solved
 
def explore(position, last_move, directions, opposite_directions, visited_positions, green_visited, junctions, image_tensor, top_left_corners,remaining_corners,exploration_visited_positions,from_junction,movement):
    
    while True:
        pos_tuple = tuple(position.tolist())
        if pos_tuple in visited_positions: #or any(pos_tuple == ep for ep in exploration_visited_positions)
            return False,remaining_corners # Avoid loops by stopping if we revisit a position
        
        whiteness_scores = {}
         # Explore directions
        for direction, (dy, dx) in directions.items():
            if direction == opposite_directions.get(last_move):
                continue  # Skip the opposite direction of the last move

            new_position = position + torch.tensor([dy, dx])  ##################here
            if (0 <= new_position[0] < image_tensor.shape[2]) and (0 <= new_position[1] < image_tensor.shape[3]):
                area = image_tensor[:, :, new_position[0]:new_position[0]+2, new_position[1]:new_position[1]+2]
                avg_whiteness = area.mean().item()
                if avg_whiteness > 0.65:
                    whiteness_scores[direction] = avg_whiteness

        
        matching_junction = next((j for j in junctions if j[0].equal(position)), None)
        if matching_junction and from_junction==True:
            exploration_visited_positions.difference_update(set(movement))
            for direction_value in matching_junction[1]:
                if direction_value[1] in whiteness_scores.values():
                    matching_junction[1].remove(direction_value)
                    break
            return False,remaining_corners

        movement.append(pos_tuple)
        
        for idx, gp_tuple in enumerate(remaining_corners):
                if abs(pos_tuple[0] - gp_tuple[0]) == 0 and abs(pos_tuple[1] - gp_tuple[1]) == 0:
                    remaining_corners.pop(idx)  # Remove the element by index
                    exploration_visited_positions.update(set(movement))
                    visited_positions.update(exploration_visited_positions) 
                    movement=[]
                    from_junction=False
                    break  

        if len(remaining_corners) == 0:
                return True,remaining_corners  

        if not whiteness_scores:
            return False,remaining_corners
        

        sorted_directions = sorted(whiteness_scores.items(),  key=lambda item: item[1], reverse=True)
        best_direction,_ = sorted_directions[0]

        if len(sorted_directions) > 1:
            junctions.append((position.clone(), sorted_directions))
            from_junction=True
            exploration_visited_positions.update(set(movement))
            movement=[]

                

        # Move in the best direction
        dy, dx = directions[best_direction]
        intermediate_position = position + torch.tensor([dy, dx])  # Calculate intermediate position hereeeeeee
        next_position = position + torch.tensor([2*dy, 2*dx])######################here
        if  next_position not in exploration_visited_positions:
            movement.append(tuple(intermediate_position.tolist()))
            position += torch.tensor([2*dy, 2*dx])# hereeeee
            last_move = best_direction
               
        else:
            return False,remaining_corners


def combine_grid(chunks,numchunks ,overlap, elimination_height=1):
    # Determine the dimensions of the chunks
    
    batch_size, iterations, _, chunk_height, chunk_width = chunks[0].shape
    stride_height = chunk_height - overlap
    stride_width = chunk_width - overlap

    if overlap >= 3:
        elimination_height = 1
    else:
        elimination_height = 0

    # Create a tensor to hold the combined output
    output_height = stride_height * 2 + overlap
    output_width = stride_width * 2 + overlap
    output_grid = torch.zeros((batch_size, iterations, 2, output_height, output_width), device=chunks[0].device)

    # Define positions for the chunks in the 2x2 grid
    positions = [
        (0, 0),  # Top-left
        (0, stride_width),  # Top-right
        (stride_height, 0),  # Bottom-left
        (stride_height, stride_width)  # Bottom-right
    ]

    for i, (start_h, start_w) in enumerate(positions):
        chunk = chunks[i]

        # Set the elimination_height rows and columns to zeros
        if elimination_height > 0:
            chunk[:, :, :, :elimination_height, :] = 0
            chunk[:, :, :, -elimination_height:, :] = 0
            chunk[:, :, :, :, :elimination_height] = 0
            chunk[:, :, :, :, -elimination_height] = 0

        output_grid[:, :, :, start_h:start_h+chunk_height, start_w:start_w+chunk_width]=torch.max(output_grid[:, :, :, start_h:start_h+chunk_height, start_w:start_w+chunk_width] , chunk)


    return output_grid

def combine_chunks(outputs, num_chunks, overlap, out_width):
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
            output_grid[:, :, 0:chunk_height-1, :] = chunk[:, :, 0:chunk_height-1, :]

        elif i == num_chunks - 1:
            output_grid[:, :, start_idx+1:, :] = chunk[:, :, 1:chunk_height, :]

        else:
            output_grid[:, :, start_idx+1:start_idx+chunk_height-1, :] = chunk[:, :, 1:chunk_height-1, :]

            
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


def split_image_into_grid(images, num_chunks,overlap=0):
    """
    Splits the image into a grid of num_chunks x num_chunks chunks.

    Parameters:
    images (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
    num_chunks (int): The number of chunks per dimension.

    Returns:
    List[torch.Tensor]: A list of image chunks.
    """
    _, _, height, width = images.shape
    # Calculate the effective chunk dimensions including overlap
    chunk_height = (height + (num_chunks - 1) * overlap) // num_chunks
    chunk_width = (width + (num_chunks - 1) * overlap) // num_chunks

    chunks = []

    for i in range(num_chunks):
        for j in range(num_chunks):
            start_height = i * (chunk_height - overlap)
            end_height = start_height + chunk_height
            start_width = j * (chunk_width - overlap)
            end_width = start_width + chunk_width

            # Handle boundary conditions to avoid index errors
            end_height = min(end_height, height)
            end_width = min(end_width, width)

            # Crop the chunk from the image
            chunk = images[:, :, max(start_height, 0):end_height, max(start_width, 0):end_width]
            chunks.append(chunk)

    return chunks
   

# def warm_up_all_convolutions(numchunks, gpu_count, device, net, overlap):
#     width = net.width
#     x_size = (1, 3, 48, 48)  # Adjust this to match your input size
#     x = torch.randn(x_size).to(device)
#     chunks = split_image_into_chunks(x, numchunks, overlap)
#     chunks_recall = torch.randn(numchunks, 1, 3, chunks.size(3), chunks.size(4)).to(device)
#     chunks_interims = torch.randn(numchunks, 1, width, chunks.size(3), chunks.size(4)).to(device)
#     chunks_head2 = torch.randn(numchunks, 1, 32, chunks.size(3), chunks.size(4)).to(device)
#     chunks_head3 = torch.randn(numchunks, 1, 8, chunks.size(3), chunks.size(4)).to(device)

#     # Define the warm-up process for each convolution layer
#     def warm_up_layer(forward_function, chunks, chunks_interims=None):
#         futures = []
#         for i in range(numchunks):
#             chunk_device = f'cuda:{i % gpu_count}' if gpu_count > 1 else device
#             chunk = chunks[i].to(chunk_device)
#             net.to(chunk_device)
            
#             if chunks_interims is not None:
#                 interim = chunks_interims[i].to(chunk_device)
#                 futures.append(torch.jit.fork(forward_function, chunk, interim))
#             else:
#                 futures.append(torch.jit.fork(forward_function, chunk))
                
#         for future in futures:
#             torch.jit.wait(future)

#     # Warm-up for all layers
#     warm_up_layer(net.special_forward_proj, chunks)
#     warm_up_layer(net.special_forward_conv_recall, chunks_recall, chunks_interims)
#     warm_up_layer(net.special_forward_recur_block_0_1, chunks_interims)
#     warm_up_layer(net.special_forward_recur_block_0_2, chunks_interims)
#     warm_up_layer(net.special_forward_recur_block_1_1, chunks_interims)
#     warm_up_layer(net.special_forward_recur_block_1_2, chunks_interims)
#     warm_up_layer(net.special_forward_head1, chunks_interims)
#     warm_up_layer(net.special_forward_head2, chunks_head2)
#     warm_up_layer(net.special_forward_head3, chunks_head3)


        
    