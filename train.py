import os
import sys
import datetime
import argparse
import logging
import logging.config
import coloredlogs
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from model import PoseNet
from dataset import H36M_Dataset
from evaluate import h36m_evaluate
from utils import mkdir, AverageMeter


config = {
    "exp_root": "./log",

    "cameras_path": "data/cameras.h5",
    "bases_path": "data/bases.npy",

    "bases_to_use": "dct",
    "input_data": "det",
    "n_bases": 8,
    "n_frames": 50,

    "window_slide": 5,
    "n_joints": 17,

    "num_workers": 8,
    "gpus": [0],
    "batch_size_per_gpu": 256,

    "train": {
        "init_lr": 1e-4,
        "lr_decay": 0.1,
        "num_epochs": 100,
        "log_per_n_iterations": 10,
    },

    "args": {},
}


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bases", type=str, help="bases to use (dct or svd)")
    parser.add_argument("-i", "--input", type=str, help="input data to the model (det or gt)")
    parser.add_argument("-f", "--nframes", type=int, help="number of frames")
    parser.add_argument("-k", "--nbases", type=int, help="number of bases")
    parser.add_argument("-g", "--gpu", type=int, help="gpu to use")
    args, _ = parser.parse_known_args()
    return args


def save_ckpt(model, exp_path):
    save_path = os.path.join(exp_path, "ckpt.pth.tar")

    torch.save({
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    }, save_path)

    config["logger"].info("Save model to {}".format(save_path))


def main():
    cl_args = parse_command_line()
    config["args"].update(vars(cl_args))

    # override config with command line args
    if config["args"]["bases"] is not None:
        assert config["args"]["bases"] in ["dct", "svd"], "Invalid bases: {}".format(config["args"]["bases"])
        config["bases_to_use"] = config["args"]["bases"]
    if config["args"]["input"] is not None:
        assert config["args"]["input"] in ["det", "gt"], "Invalid input: {}".format(config["args"]["input"])
        config["input_data"] = config["args"]["input"]
    if config["args"]["nframes"] is not None:
        config["n_frames"] = config["args"]["nframes"]
    if config["args"]["nbases"] is not None:
        config["n_bases"] = config["args"]["nbases"]
    if config["args"]["gpu"] is not None:
        config["gpus"] = [config["args"]["gpu"]]

    # exp folder
    exp_tag = "{}_F{}_k{}_{}_{}".format(datetime.datetime.now().strftime("%m%d_%H%M%S"), config["n_frames"], config["n_bases"], config["bases_to_use"], config["input_data"])
    exp_path = os.path.join(config["exp_root"], exp_tag)
    config["exp_tag"] = exp_tag
    config["exp_path"] = exp_path
    mkdir(exp_path)

    # logger
    logger = logging.getLogger()
    coloredlogs.install(level="DEBUG", logger=logger)
    fileHandler = logging.FileHandler(os.path.join(exp_path, "log.txt"))
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    config["logger"] = logger

    logger.info(sys.argv)

    # setup gpus
    gpus = ','.join([str(x) for x in config["gpus"]])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info("Set CUDA_VISIBLE_DEVICES to {}".format(gpus))

    # model
    model = PoseNet(config)
    model = nn.DataParallel(model)
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["init_lr"])

    # load bases
    if config["bases_to_use"] == "svd":
        fixed_bases = np.load(config["bases_path"])
        assert config["n_bases"] <= fixed_bases.shape[0] and config["n_frames"] == fixed_bases.shape[1], fixed_bases.shape
        fixed_bases = fixed_bases[:config["n_bases"]]
        # scale svd bases to the same magnitude as dct bases
        # the scaling factor here is for F=50
        fixed_bases *= np.sqrt(25)
    elif config["bases_to_use"] == "dct":
        x = np.arange(config["n_frames"])
        fixed_bases = [np.ones([config["n_frames"]]) * np.sqrt(0.5)]
        for i in range(1, config["n_bases"]):
            fixed_bases.append(np.cos(i * np.pi * ((x + 0.5) / config["n_frames"])))
        fixed_bases = np.array(fixed_bases)
    else:
        assert False, config["bases_to_use"]
    config["bases"] = fixed_bases

    fixed_bases = torch.from_numpy(fixed_bases).float()  # (K, F)
    fixed_bases = fixed_bases.view(1, config["n_bases"], config["n_frames"])  # (1, K, F)

    # dataset & dataloader
    train_dataset = H36M_Dataset(config, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size_per_gpu"]*len(config["gpus"]), shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    test_dataset = H36M_Dataset(config, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size_per_gpu"]*len(config["gpus"]), shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    tot_step = 0

    for epoch in range(config["train"]["num_epochs"]):

        # learning rate decay
        if epoch in [60, 85]:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * config["train"]["lr_decay"]
            logger.info("Learning rate set to {}".format(optimizer.param_groups[0]["lr"]))

        # train one epoch
        model.train()

        for step, batch in enumerate(train_loader):

            # parse batch data
            data_2d_gt = batch["data_2d_gt"]  # (B, Jx2, F)
            data_2d_cpn = batch["data_2d_cpn"]  # (B, Jx2, F)
            if config["input_data"] == "det":
                data_2d = data_2d_cpn
            elif config["input_data"] == "gt":
                data_2d = data_2d_gt
            else:
                assert False, config["input_data"]
            data_3d = batch["data_3d"]  # (B, Jx3, F)
            mean_3d = batch["mean_3d"]  # (B, Jx3)
            std_3d = batch["std_3d"]  # (B, Jx3)

            B = data_3d.shape[0]
            batch_bases = fixed_bases.repeat(B, 1, 1)  # (B, K, F)

            data_2d = data_2d.cuda()
            data_3d = data_3d.cuda()
            batch_bases = batch_bases.cuda()
            mean_3d = mean_3d.cuda()
            std_3d = std_3d.cuda()

            # forward pass
            coeff = model(data_2d, batch_bases)

            # compute loss
            loss = model.module.build_loss_training(coeff, batch_bases, data_3d, mean_3d, std_3d)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_step += 1

            if "log_per_n_iterations" in config["train"] and (step + 1) % config["train"]["log_per_n_iterations"] == 0:
                logger.info("TRAIN Epoch {}, step {}/{} ({}): loss = {:.6f}".format(
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    tot_step,
                    loss.item()))

        # testing
        model.eval()
        logger.info("Testing on test set...")

        total_loss = AverageMeter()

        gts_3d = []
        preds_3d = []
        indices = []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):

                # parse batch data
                data_2d_gt = batch["data_2d_gt"]  # (B, Jx2, F)
                data_2d_cpn = batch["data_2d_cpn"]  # (B, Jx2, F)
                if config["input_data"] == "det":
                    data_2d = data_2d_cpn
                elif config["input_data"] == "gt":
                    data_2d = data_2d_gt
                else:
                    assert False, config["input_data"]
                data_2d_gt_flip = batch["data_2d_gt_flip"]  # (B, Jx2, F)
                data_2d_cpn_flip = batch["data_2d_cpn_flip"]  # (B, Jx2, F)
                if config["input_data"] == "det":
                    data_2d_flip = data_2d_cpn_flip
                elif config["input_data"] == "gt":
                    data_2d_flip = data_2d_gt_flip
                else:
                    assert False, config["input_data"]
                data_3d = batch["data_3d"]  # (B, Jx3, F)
                data_3d_flip = batch["data_3d_flip"]  # (B, Jx3, F)
                mean_3d = batch["mean_3d"]  # (B, Jx3)
                std_3d = batch["std_3d"]  # (B, Jx3)
                idx = batch["idx"]  # (B,)

                B = data_3d.shape[0]
                batch_bases = fixed_bases.repeat(B, 1, 1)  # (B, K, F)

                data_2d = data_2d.cuda()
                data_2d_flip = data_2d_flip.cuda()
                data_3d = data_3d.cuda()
                data_3d_flip = data_3d_flip.cuda()
                batch_bases = batch_bases.cuda()
                mean_3d = mean_3d.cuda()
                std_3d = std_3d.cuda()

                # forward pass
                coeff = model(data_2d, batch_bases)
                coeff_flip = model(data_2d_flip, batch_bases)

                # compute loss
                loss, res = model.module.build_loss_test((coeff, coeff_flip), batch_bases, (data_3d, data_3d_flip), mean_3d, std_3d)
                pred_3d, gt_3d = res

                total_loss.add(loss.item())

                preds_3d.append(pred_3d)
                gts_3d.append(gt_3d)
                indices.append(idx.data.numpy())

        avg_loss = total_loss.value()
        logger.info("Test loss: {}".format(avg_loss))

        if epoch == config["train"]["num_epochs"] - 1:
            preds_3d = np.concatenate(preds_3d, 0)
            gts_3d = np.concatenate(gts_3d, 0)
            indices = np.concatenate(indices, 0)
            h36m_evaluate(preds_3d, gts_3d, indices, test_dataset, config)
            save_ckpt(model, exp_path)


if __name__ == "__main__":
    main()
