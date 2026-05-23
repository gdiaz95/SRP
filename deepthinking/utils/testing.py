""" testing.py
    Utilities for testing models
"""
import os
import errno

import einops
import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from torchvision.utils import save_image

# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


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


def test(net, loaders, mode, iters, problem, device, numchunks, overlap, terminals,
         parallel, termination, time_eval, acc_eval,
         tc_threshold=0.65, tc_first_batch=20):
    accs = []
    for loader in loaders:
        if termination and not parallel:
            accuracy = test_default_TC(net, loader, iters, device, terminals,
                                       time_eval, acc_eval, tc_threshold, tc_first_batch)
        elif not termination and not parallel:
            accuracy = test_default(net, loader, iters, problem, device, terminals,
                                    time_eval, acc_eval)
        elif mode == "default" and parallel:
            accuracy = test_default_parallel(net, loader, iters, device, numchunks,
                                             overlap, terminals, time_eval, acc_eval)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


def test_default(net, testloader, iters, problem, device, terminals, time_eval, acc_eval):
    max_iters = 70 if acc_eval else (50 if time_eval else max(iters))

    net.eval()
    corrects = torch.zeros(max_iters)
    durations = []
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            import time
            start_time = time.time()
            all_outputs, _ = net(inputs, iters_to_do=max_iters)
            end_time = time.time()
            duration = ((end_time - start_time) / all_outputs.shape[0]) / max_iters
            durations.append(duration)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    ret_acc = {}

    if time_eval:
        np.save(f"no_TC_{terminals}_G_{max_iters}_iters.npy", durations)
    if acc_eval:
        np.save(f"no_TC_accuracy_{terminals}G.npy", np.array(accuracy))

    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc


def test_default_TC(net, testloader, iters, device, terminals, time_eval, acc_eval,
                    tc_threshold=0.65, tc_first_batch=20):
    """Test with the Termination Condition (TC) module (Algorithm 1 in paper).

    tc_threshold: path-intensity threshold for the TC traversal (0.65 for synthetic, 0.50 for ZInD).
    tc_first_batch: RB iterations in the first batch (20 for synthetic, 10 for ZInD).
    """
    max_iters = 20 if acc_eval else (4 if time_eval else int(max(iters) / 10))

    net.eval()
    total = 0
    corrects_per_iteration = torch.zeros(max_iters * 10)
    mistakes = []
    durations = []

    import time

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(testloader, leave=False), start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            corrects = torch.zeros(targets.size(0), max_iters * 10)
            predicted_targets = torch.ones(targets.size(), device=device)
            interims = None
            first = True

            for i in range(max_iters + 1):
                start_time_total = time.time()
                iters_this_batch = tc_first_batch if first else 10
                all_outputs, interims = net(inputs, iters_this_batch, interims)
                first = False

                single_image = apply_mask(inputs, all_outputs)
                single_target, finishidx, _ = check_termination_multiple(
                    single_image, tc_threshold
                )
                predicted_targets[:, :] = single_target.squeeze()

                if i == max_iters and not finishidx:
                    duration_total = time.time() - start_time_total
                    durations.append(duration_total)
                    print(f"[MISTAKE] sample {j}: TC never converged after {max_iters} rounds")
                    mistakes.append((j, inputs.cpu().numpy()[0],
                                     targets.cpu().numpy()[0],
                                     single_image.cpu().numpy()[0]))
                    break

                if finishidx:
                    duration_total = time.time() - start_time_total
                    durations.append(duration_total)
                    num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                    num_ones = targets.sum(dim=[1, 2]).to(device)
                    correct = int(num_ones_predicted <= (num_ones + 8))
                    corrects[:, i:max_iters * 10] = correct
                    if not correct:
                        print(f"[MISTAKE] sample {j}: wrong answer at TC round {i} "
                              f"(predicted {int(num_ones_predicted)} path pixels, "
                              f"target {int(num_ones[0])})")
                        mistakes.append((j, inputs.cpu().numpy()[0],
                                         targets.cpu().numpy()[0],
                                         single_image.cpu().numpy()[0]))
                    break

            for i in range(max_iters * 10):
                corrects_per_iteration[i] += corrects[:, i].sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects_per_iteration / total

    np.save(f"mistakes_{terminals}G.npy", np.array(mistakes, dtype=object))

    if time_eval:
        np.save(f"TC_{terminals}_G_{max_iters * 10}_iters.npy", durations)
    if acc_eval:
        np.save(f"TC_accuracy_{terminals}G.npy", np.array(accuracy))

    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    ret_acc[-1] = accuracy[-1].item()
    return ret_acc


def test_default_parallel(net, testloader, iters, device, numchunks, overlap,
                          terminals, time_eval, acc_eval):
    """Test using the parallelised MazeNet forward pass (Appendix A.4 in paper)."""
    max_iters = max(iters)
    net.eval()
    corrects_per_iteration = torch.zeros(max_iters)
    iters_to_do = 10
    total = 0
    durations = []

    import time

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            corrects = torch.zeros(targets.size(0), max_iters)
            interims = None

            for i in range(iters_to_do, max_iters + 1, iters_to_do):
                start_time_total = time.time()
                single_image, interims = net.forward_parallel1_1gpu(
                    inputs, iters_to_do, numchunks, overlap, device, interims
                )
                single_target, finishidx, _ = check_termination_multiple(single_image)

                if i == max_iters and not finishidx:
                    durations.append(time.time() - start_time_total)
                    break
                if finishidx:
                    durations.append(time.time() - start_time_total)
                    num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                    num_ones = targets.sum(dim=[1, 2]).to(device)
                    correct = int(num_ones_predicted <= (num_ones + 8))
                    if correct:
                        corrects[:, i:max_iters] = 1
                    break

            for i in range(max_iters):
                corrects_per_iteration[i] += corrects[:, i].sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects_per_iteration / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc


def apply_mask(inputs, all_outputs):
    channel_sum = torch.sum(inputs, dim=1)
    just_path_single = (channel_sum > 0).float().unsqueeze(1).expand(-1, 3, -1, -1)
    single_predicted = all_outputs[:, -1, 1]
    reshape_single = (single_predicted
                      .view(inputs.size(0), inputs.shape[2], inputs.shape[3])
                      .unsqueeze(1).expand(-1, 3, -1, -1))
    overlayed = reshape_single * inputs
    green_mask = ((inputs[:, 1, :, :] == 1) &
                  (inputs[:, 0, :, :] == 0) &
                  (inputs[:, 2, :, :] == 0))
    overlayed[:, 1, :, :].masked_fill_(green_mask, 1)
    overlayed = overlayed * inputs
    return torch.clamp(overlayed + just_path_single * 0.2, 0, 1)


def Fill_target_multiple(single_output, visited_positions):
    visited_positions = torch.tensor(visited_positions)
    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    all_positions = (visited_positions[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    target_tensor = torch.zeros((1, 1, single_output.shape[2], single_output.shape[3]))
    y, x = all_positions[:, 0], all_positions[:, 1]
    target_tensor[0, 0, y, x] = 1
    return target_tensor


def check_termination_multiple(single_output, tc_threshold=0.65):
    green_channel = single_output[:, 1, :, :]
    red_channel = single_output[:, 0, :, :]
    blue_channel = single_output[:, 2, :, :]
    is_green = ((green_channel > 0.8) &
                (red_channel < 0.5) &
                (blue_channel < 0.5))
    green_pixel_positions = torch.nonzero(is_green)
    top_left_corners = identify_top_left_corners(green_pixel_positions)
    visited_positions, check = move_towards_whiteness_check_green_multiple(
        single_output, top_left_corners, tc_threshold
    )
    single_target = Fill_target_multiple(single_output, visited_positions)
    return single_target, check, visited_positions


def identify_top_left_corners(green_pixel_positions):
    green_pixel_positions = green_pixel_positions.clone().detach()
    yx_positions = green_pixel_positions[:, 1:]
    max_y, max_x = yx_positions.max(0)[0]
    grid = torch.zeros((max_y + 2, max_x + 2), dtype=torch.bool)
    grid[yx_positions[:, 0], yx_positions[:, 1]] = True
    top_left_corners = []
    for y, x in yx_positions:
        if grid[y, x]:
            if grid[y, x] & grid[y, x + 1] & grid[y + 1, x] & grid[y + 1, x + 1]:
                top_left_corners.append([y.item(), x.item()])
                grid[y:y + 2, x:x + 2] = False
    return torch.tensor(top_left_corners, device=green_pixel_positions.device)


def move_towards_whiteness_check_green_multiple(image_tensor, top_left_corners, tc_threshold=0.65):
    directions = {'east': (0, 2), 'south': (2, 0), 'north': (-2, 0), 'west': (0, -2)}
    opposite_directions = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}

    start = top_left_corners[0]
    current_position = torch.tensor([start[0].item(), start[1].item()])
    visited_positions = set()
    green_visited = set()
    junctions = []
    exploration_visited_positions = set()
    from_junction = False
    movement = []

    solved, remaining_corners = explore(
        current_position, None, directions, opposite_directions,
        visited_positions, green_visited, junctions, image_tensor,
        top_left_corners, top_left_corners.tolist(),
        exploration_visited_positions, from_junction, movement, tc_threshold
    )

    while junctions and len(remaining_corners) > 0:
        position, unexplored_directions = junctions.pop()
        for direction_name, _ in unexplored_directions[1:]:
            new_dy, new_dx = directions[direction_name]
            new_position = position + torch.tensor([2 * new_dy, 2 * new_dx])
            intermediate_position = position + torch.tensor([new_dy, new_dx])
            if tuple(new_position.tolist()) not in visited_positions:
                movement = [tuple(intermediate_position.tolist())]
                from_junction = True
                solved, remaining_corners = explore(
                    new_position, direction_name, directions, opposite_directions,
                    visited_positions, green_visited, junctions, image_tensor,
                    top_left_corners, remaining_corners,
                    exploration_visited_positions, from_junction, movement, tc_threshold
                )

    return list(visited_positions), solved


def explore(position, last_move, directions, opposite_directions, visited_positions,
            green_visited, junctions, image_tensor, top_left_corners, remaining_corners,
            exploration_visited_positions, from_junction, movement, tc_threshold=0.65):

    while True:
        pos_tuple = tuple(position.tolist())
        if pos_tuple in visited_positions:
            return False, remaining_corners

        whiteness_scores = {}
        for direction, (dy, dx) in directions.items():
            if direction == opposite_directions.get(last_move):
                continue
            new_position = position + torch.tensor([dy, dx])
            if (0 <= new_position[0] < image_tensor.shape[2] and
                    0 <= new_position[1] < image_tensor.shape[3]):
                area = image_tensor[
                    :, :,
                    new_position[0]:new_position[0] + 2,
                    new_position[1]:new_position[1] + 2
                ]
                avg_whiteness = area.mean().item()
                if avg_whiteness > tc_threshold:
                    whiteness_scores[direction] = avg_whiteness

        matching_junction = next((j for j in junctions if j[0].equal(position)), None)
        if matching_junction and from_junction:
            exploration_visited_positions.difference_update(set(movement))
            for direction_value in matching_junction[1]:
                if direction_value[1] in whiteness_scores.values():
                    matching_junction[1].remove(direction_value)
                    break
            return False, remaining_corners

        movement.append(pos_tuple)

        for idx, gp_tuple in enumerate(remaining_corners):
            if abs(pos_tuple[0] - gp_tuple[0]) == 0 and abs(pos_tuple[1] - gp_tuple[1]) == 0:
                remaining_corners.pop(idx)
                exploration_visited_positions.update(set(movement))
                visited_positions.update(exploration_visited_positions)
                movement = []
                from_junction = False
                break

        if len(remaining_corners) == 0:
            return True, remaining_corners

        if not whiteness_scores:
            return False, remaining_corners

        sorted_directions = sorted(whiteness_scores.items(), key=lambda item: item[1], reverse=True)
        best_direction, _ = sorted_directions[0]

        if len(sorted_directions) > 1:
            junctions.append((position.clone(), sorted_directions))
            from_junction = True
            exploration_visited_positions.update(set(movement))
            movement = []

        dy, dx = directions[best_direction]
        intermediate_position = position + torch.tensor([dy, dx])
        next_position = position + torch.tensor([2 * dy, 2 * dx])
        if next_position not in exploration_visited_positions:
            movement.append(tuple(intermediate_position.tolist()))
            position += torch.tensor([2 * dy, 2 * dx])
            last_move = best_direction
        else:
            return False, remaining_corners


def combine_chunks(outputs, num_chunks, overlap, out_width):
    num_chunks = outputs.shape[0]
    batch_size, _, chunk_height, width = outputs[0].shape
    stride = chunk_height - overlap
    output_height = stride * (num_chunks - 1) + chunk_height
    output_grid = torch.zeros((batch_size, out_width, output_height, width),
                              device=outputs[0].device)
    for i in range(num_chunks):
        chunk = outputs[i]
        start_idx = i * stride
        if i == 0:
            output_grid[:, :, 0:chunk_height - 1, :] = chunk[:, :, 0:chunk_height - 1, :]
        elif i == num_chunks - 1:
            output_grid[:, :, start_idx + 1:, :] = chunk[:, :, 1:chunk_height, :]
        else:
            output_grid[:, :, start_idx + 1:start_idx + chunk_height - 1, :] = (
                chunk[:, :, 1:chunk_height - 1, :]
            )
    return output_grid


def split_image_into_chunks(images, num_chunks, overlap=0):
    batch_size, channels, height, width = images.shape
    total_overlap = (num_chunks - 1) * overlap
    chunk_height = (height + total_overlap) // num_chunks
    concatenated_chunks = torch.empty(num_chunks, batch_size, channels, chunk_height, width,
                                      device=images.device)
    for i in range(num_chunks):
        start_index = i * (chunk_height - overlap)
        end_index = start_index + chunk_height
        concatenated_chunks[i, :, :, 0:chunk_height, :] = images[:, :, start_index:end_index, :]
    return concatenated_chunks
