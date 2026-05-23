""" train_model.py
    Train, test, and save models.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import json
import logging
import os
import re
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import deepthinking as dt
import deepthinking.utils.logging_utils as lg

import matplotlib.pyplot as plt

# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def plot_and_save_test_acc_over_epochs(test_accuracies, test_iterations, save_path):
    plt.figure()
    for epoch, epoch_accs in enumerate(test_accuracies):
        accs = [epoch_accs[iteration] for iteration in test_iterations]
        plt.plot(test_iterations, accs, label=f"Epoch {epoch}")
    plt.xlabel("Test Iterations")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy over Iterations")
    plt.legend()
    plt.savefig(os.path.join(save_path, "test_accuracy_over_epochs.png"))
    plt.close()
    np.save(os.path.join(save_path, "test_accuracies.npy"), np.array(test_accuracies))
    np.save(os.path.join(save_path, "test_iterations.npy"), np.array(test_iterations))


@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    cfg.problem.model.test_iterations = list(range(
        cfg.problem.model.test_iterations["low"],
        cfg.problem.model.test_iterations["high"] + 1,
    ))
    assert 0 <= cfg.problem.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."
    writer = SummaryWriter(
        log_dir=f"tensorboard-{cfg.problem.model.model}-{cfg.problem.hyp.alpha}"
    )
    terminals = int(re.search(r"\d+", cfg.problem.data_type).group())

    loaders = dt.utils.get_dataloaders(cfg.problem)

    if cfg.retrain:
        cfg.problem.model.model_path = cfg.model_path

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(
        cfg.problem.name, cfg.problem.model, device
    )
    if cfg.retrain:
        start_epoch = 0

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer(
        cfg.problem.hyp, cfg.problem.model, net, optimizer_state_dict
    )
    train_setup = dt.TrainingSetup(
        optimizer=optimizer,
        scheduler=lr_scheduler,
        warmup=warmup_scheduler,
        clip=cfg.problem.hyp.clip,
        alpha=cfg.problem.hyp.alpha,
        max_iters=cfg.problem.model.max_iters,
        problem=cfg.problem.name,
    )

    log.info(f"==> Starting training for {max(cfg.problem.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    best_so_far = False

    test_accuracies = []
    test_iterations = list(range(2, 70))

    for epoch in range(start_epoch, cfg.problem.hyp.epochs):
        loss, acc = dt.train(net, loaders, cfg.problem.hyp.train_mode, train_setup, device)

        # Evaluate without TC during training (fast, GPU-only)
        epoch_test_accs = dt.test(
            net, [loaders["test"]], cfg.problem.hyp.test_mode, test_iterations,
            cfg.problem.name, device, cfg.problem.numchunks, cfg.problem.overlap,
            terminals, cfg.problem.parallel_testing,
            False, False, False,  # termination_condition, time_eval, acc_eval
        )[0]
        test_accuracies.append(epoch_test_accs)
        val_acc = np.max(list(epoch_test_accs.values()))
        val_acc_index = list(epoch_test_accs.values()).index(val_acc) + 2

        if val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = val_acc

        log.info(f"Training loss at epoch {epoch}: {loss}")
        log.info(f"Training accuracy at epoch {epoch}: {acc}")
        log.info(f"Val accuracy at epoch {epoch}: {val_acc} at iteration {val_acc_index}")

        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)
        writer.add_scalar("Accuracy/val_acc", val_acc, epoch)
        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}", optimizer.param_groups[i]["lr"], epoch)

        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_accuracies.npy"),
                np.array(test_accuracies))
        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_iterations.npy"),
                np.array(test_iterations))

        if (epoch + 1) % cfg.problem.hyp.val_period == 0 or epoch + 1 == cfg.problem.hyp.epochs:
            test_acc, val_acc, train_acc = dt.test(
                net,
                [loaders["test"], loaders["val"], loaders["train"]],
                cfg.problem.hyp.test_mode,
                cfg.problem.model.test_iterations,
                cfg.problem.name, device,
                cfg.problem.numchunks, cfg.problem.overlap,
            )
            log.info(f"Training accuracy: {train_acc}")
            log.info(f"Val accuracy: {val_acc}")
            log.info(f"Test accuracy (hard data): {test_acc}")

            tb_last = cfg.problem.model.test_iterations[-1]
            lg.write_to_tb(
                [train_acc[tb_last], val_acc[tb_last], test_acc[tb_last]],
                ["train_acc", "val_acc", "test_acc"],
                epoch, writer,
            )

        save_now = (
            (epoch + 1) % cfg.problem.hyp.save_period == 0
            or (epoch + 1) == cfg.problem.hyp.epochs
            or best_so_far
        )
        if save_now:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
            out_str = f"model_{'best' if best_so_far else ''}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)

    plot_and_save_test_acc_over_epochs(
        test_accuracies, test_iterations,
        os.path.dirname(os.path.realpath(__file__))
    )

    writer.flush()
    writer.close()

    stats = OrderedDict([
        ("max_iters", cfg.problem.model.max_iters),
        ("run_id", cfg.run_id),
        ("test_acc", test_acc),
        ("test_data", cfg.problem.test_data),
        ("test_iters", list(cfg.problem.model.test_iterations)),
        ("test_mode", cfg.problem.hyp.test_mode),
        ("train_data", cfg.problem.train_data),
        ("train_acc", train_acc),
        ("val_acc", val_acc),
    ])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)


if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
