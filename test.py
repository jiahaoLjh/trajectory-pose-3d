import os
import glob
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

    "num_workers": 0,
    "gpus": [0],
    "batch_size_per_gpu": 256,

    "args": {},
}


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", required=True, type=str, help="experiment to load")
    parser.add_argument("-g", "--gpu", type=int, help="gpu to use")
    args, _ = parser.parse_known_args()
    return args


def load_ckpt(model, exp_path):
    load_path = os.path.join(exp_path, "ckpt.pth.tar")
    ckpt = torch.load(load_path)
    config["logger"].info("Load model from {}".format(load_path))

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)


def main():
    cl_args = parse_command_line()
    config["args"].update(vars(cl_args))

    # override config with command line args
    if config["args"]["gpu"] is not None:
        config["gpus"] = [config["args"]["gpu"]]

    # exp folder
    dir_name = glob.glob(os.path.join(config["exp_root"], "{}*".format(config["args"]["exp"])))
    assert len(dir_name) == 1, "Invalid exp folder to load: {}".format(config["args"]["exp"])
    exp_tag = os.path.basename(dir_name[0])
    exp_path = os.path.join(config["exp_root"], exp_tag)
    config["exp_tag"] = exp_tag
    config["exp_path"] = exp_path

    # readout config from exp tag
    _, _, exp_f, exp_k, exp_bases, exp_input = exp_tag.split("_")
    exp_f = int(exp_f[1:])
    exp_k = int(exp_k[1:])
    config["bases_to_use"] = exp_bases
    config["input_data"] = exp_input
    config["n_bases"] = exp_k
    config["n_frames"] = exp_f

    # logger
    logger = logging.getLogger()
    coloredlogs.install(level="DEBUG", logger=logger)
    config["logger"] = logger

    # setup gpus
    gpus = ','.join([str(x) for x in config["gpus"]])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info("Set CUDA_VISIBLE_DEVICES to {}".format(gpus))

    # model
    model = PoseNet(config)
    model = nn.DataParallel(model)
    model = model.cuda()

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
    train_dataset = H36M_Dataset(config, "train")   # training set must be loaded first to compute stats
    test_dataset = H36M_Dataset(config, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size_per_gpu"]*len(config["gpus"]), shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    load_ckpt(model, exp_path)

    model.eval()
    logger.info("Inference on test set...")

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

            _, res = model.module.build_loss_test((coeff, coeff_flip), batch_bases, (data_3d, data_3d_flip), mean_3d, std_3d)
            pred_3d, gt_3d = res

            preds_3d.append(pred_3d)
            gts_3d.append(gt_3d)
            indices.append(idx.data.numpy())

    preds_3d = np.concatenate(preds_3d, 0)
    gts_3d = np.concatenate(gts_3d, 0)
    indices = np.concatenate(indices, 0)
    h36m_evaluate(preds_3d, gts_3d, indices, test_dataset, config)


if __name__ == "__main__":
    main()
