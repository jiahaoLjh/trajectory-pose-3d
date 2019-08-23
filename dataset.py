import numpy as np
import os
import h5py
import logging
import torch.utils.data

import cameras


# h36m has 32 joints, 17 of which are used.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'


class H36M_Dataset(torch.utils.data.Dataset):

    def __init__(self, config, mode):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert mode in ["train", "test"], "Invalid mode: {}".format(mode)

        self.config = config
        self.mode = mode

        subject_ids = [1, 5, 6, 7, 8, 9, 11]
        rcams = cameras.load_cameras(config["cameras_path"], subject_ids)
        self.rcams = rcams

        if os.path.isfile("data/train.h5") and os.path.isfile("data/test.h5"):
            with h5py.File("data/train.h5", "r") as f:
                train_set_3d = {}
                for k in f["data_3d"]:
                    d = f["data_3d"][k]
                    key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                    train_set_3d[key] = d[:]
                train_set_2d_gt = {}
                for k in f["data_2d_gt"]:
                    d = f["data_2d_gt"][k]
                    key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                    train_set_2d_gt[key] = d[:]
                
            with h5py.File("data/test.h5", "r") as f:
                test_set_3d = {}
                for k in f["data_3d"]:
                    d = f["data_3d"][k]
                    key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                    test_set_3d[key] = d[:]
                test_set_2d_gt = {}
                for k in f["data_2d_gt"]:
                    d = f["data_2d_gt"][k]
                    key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                    test_set_2d_gt[key] = d[:]
                
            self.logger.info("{} 3d train files, {} 3d test files are loaded.".format(len(train_set_3d), len(test_set_3d)))
            self.logger.info("{} 2d GT train files, {} 2d GT test files are loaded.".format(len(train_set_2d_gt), len(test_set_2d_gt)))
        else:
            raise Exception("Dataset file is missing!")

        f_cpn = np.load("data/data_cpn.npz")
        data_2d_cpn = f_cpn["positions_2d"].item()

        self.n_frames = config["n_frames"]
        self.n_joints = config["n_joints"]
        self.n_bases = config["n_bases"]
        self.window_slide = config["window_slide"]
        self.bases = config["bases"]

        dims_17 = np.where(np.array([x != '' for x in H36M_NAMES]))[0]

        assert self.n_joints == 17, self.n_joints
        dim_2d = np.sort(np.hstack([dims_17 * 2 + i for i in range(2)]))
        dim_3d = np.sort(np.hstack([dims_17 * 3 + i for i in range(3)]))

        self.left_right_symmetry_2d = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        self.left_right_symmetry_3d = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12])

        dim_cpn_to_gt = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        self.data_2d_gt = {}
        self.data_2d_cpn = {}
        self.data_3d = {}
        self.indices = []

        if mode == "train":
            data_3d = train_set_3d
            data_2d_gt = train_set_2d_gt
        else:
            data_3d = test_set_3d
            data_2d_gt = test_set_2d_gt

        # cut videos into short clips of fixed length
        self.logger.info("Loading sequence...")
        for idx, k in enumerate(sorted(data_3d)):
            if k[0] == 11 and k[2].split(".")[0] == "Directions":
                # one video is missing
                # drop all four videos instead of only one camera's view
                self.data_3d[k] = None
                continue

            assert k in data_2d_gt, k
            assert data_3d[k].shape[0] == data_2d_gt[k].shape[0]

            cam_name = k[2].split(".")[1]
            cam_id = cameras.cam_name_to_id[cam_name]

            d2_cpn = data_2d_cpn["S{}".format(k[0])][k[2].split(".")[0]][cam_id-1][:data_3d[k].shape[0], dim_cpn_to_gt]
            d2_cpn = d2_cpn.reshape([d2_cpn.shape[0], self.n_joints * 2])
            self.data_2d_cpn[k] = d2_cpn

            d2_gt = data_2d_gt[k][:, dim_2d]
            d2_gt = d2_gt.reshape([d2_gt.shape[0], self.n_joints, 2])
            d2_gt = d2_gt.reshape([d2_gt.shape[0], self.n_joints * 2])
            self.data_2d_gt[k] = d2_gt

            d3 = data_3d[k][:, dim_3d]
            d3 = d3.reshape([d3.shape[0], self.n_joints, 3])
            # align root to origin
            d3 = d3 - d3[:, :1, :]
            d3 = d3.reshape([d3.shape[0], self.n_joints * 3])
            # remove zero root joint
            d3 = d3[:, 3:]
            self.data_3d[k] = d3

            N = data_3d[k].shape[0]
            n = 0
            while n + self.n_frames <= N:
                self.indices.append((idx,) + k + (n, self.n_frames))

                n += self.window_slide

        self.n_data = len(self.indices)
        self.logger.info("{} data loaded for {} dataset".format(self.n_data, mode))

        # computing statistics for data normalization
        if "stats" in config:
            assert mode == "test", mode
            stats_data = config["stats"]
            self.logger.info("Loading stats...")
            self.mean_2d, self.std_2d, self.mean_3d, self.std_3d = stats_data
        else:
            assert mode == "train", mode

            self.mean_2d = np.mean(np.vstack(self.data_2d_gt.values()), axis=0)  # (2J,)
            self.std_2d = np.std(np.vstack(self.data_2d_gt.values()), axis=0)  # (2J,)
            self.mean_3d = np.mean(np.vstack(self.data_3d.values()), axis=0)  # (3J,)
            self.std_3d = np.std(np.vstack(self.data_3d.values()), axis=0)  # (3J,)

            self.logger.info("mean 2d: {}".format(self.mean_2d))
            self.logger.info("std 2d: {}".format(self.std_2d))
            self.logger.info("mean 3d: {}".format(self.mean_3d))
            self.logger.info("std 3d: {}".format(self.std_3d))

            stats_data = self.mean_2d, self.std_2d, self.mean_3d, self.std_3d
            config["stats"] = stats_data
            self.logger.info("Saving stats...")

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        assert 0 <= item < self.n_data, "Index {} out of range [{}, {})".format(item, 0, self.n_data)

        index = self.indices[item]
        idx, k0, k1, k2, n, t = index

        data_3d = self.data_3d[k0, k1, k2][n: n + t]
        data_2d_gt = self.data_2d_gt[k0, k1, k2][n: n + t]
        data_2d_cpn = self.data_2d_cpn[k0, k1, k2][n: n + t]

        # flip the training data with a probability of 0.5
        flip = np.random.random() < 0.5

        if self.mode == "train" and flip:
            data_2d_gt = np.reshape(data_2d_gt.copy(), [t, self.n_joints, 2])
            data_2d_gt[:, :, 0] *= -1
            data_2d_gt = data_2d_gt[:, self.left_right_symmetry_2d, :]
            data_2d_gt = np.reshape(data_2d_gt, [t, -1])

            data_2d_cpn = np.reshape(data_2d_cpn.copy(), [t, self.n_joints, 2])
            data_2d_cpn[:, :, 0] *= -1
            data_2d_cpn = data_2d_cpn[:, self.left_right_symmetry_2d, :]
            data_2d_cpn = np.reshape(data_2d_cpn, [t, -1])

            data_3d = np.reshape(data_3d.copy(), [t, self.n_joints-1, 3])
            data_3d[:, :, 0] *= -1
            data_3d = data_3d[:, self.left_right_symmetry_3d, :]
            data_3d = np.reshape(data_3d, [t, -1])

        if self.mode == "test":
            data_2d_gt_flip = np.reshape(data_2d_gt.copy(), [t, self.n_joints, 2])
            data_2d_gt_flip[:, :, 0] *= -1
            data_2d_gt_flip = data_2d_gt_flip[:, self.left_right_symmetry_2d, :]
            data_2d_gt_flip = np.reshape(data_2d_gt_flip, [t, -1])

            data_2d_cpn_flip = np.reshape(data_2d_cpn.copy(), [t, self.n_joints, 2])
            data_2d_cpn_flip[:, :, 0] *= -1
            data_2d_cpn_flip = data_2d_cpn_flip[:, self.left_right_symmetry_2d, :]
            data_2d_cpn_flip = np.reshape(data_2d_cpn_flip, [t, -1])

            data_3d_flip = np.reshape(data_3d.copy(), [t, self.n_joints - 1, 3])
            data_3d_flip[:, :, 0] *= -1
            data_3d_flip = data_3d_flip[:, self.left_right_symmetry_3d, :]
            data_3d_flip = np.reshape(data_3d_flip, [t, -1])

        data_2d_gt = (data_2d_gt - self.mean_2d) / self.std_2d
        data_2d_cpn = (data_2d_cpn - self.mean_2d) / self.std_2d
        data_3d = (data_3d - self.mean_3d) / self.std_3d

        data_2d_gt = torch.from_numpy(data_2d_gt.transpose((1, 0))).float()
        data_2d_cpn = torch.from_numpy(data_2d_cpn.transpose((1, 0))).float()
        data_3d = torch.from_numpy(data_3d.transpose((1, 0))).float()

        mean_3d = torch.from_numpy(self.mean_3d).float()
        std_3d = torch.from_numpy(self.std_3d).float()

        if self.mode == "test":
            data_2d_gt_flip = (data_2d_gt_flip - self.mean_2d) / self.std_2d
            data_2d_cpn_flip = (data_2d_cpn_flip - self.mean_2d) / self.std_2d
            data_3d_flip = (data_3d_flip - self.mean_3d) / self.std_3d

            data_2d_gt_flip = torch.from_numpy(data_2d_gt_flip.transpose((1, 0))).float()
            data_2d_cpn_flip = torch.from_numpy(data_2d_cpn_flip.transpose((1, 0))).float()
            data_3d_flip = torch.from_numpy(data_3d_flip.transpose((1, 0))).float()

        ret = {
            "data_2d_gt": data_2d_gt,
            "data_2d_cpn": data_2d_cpn,
            "data_3d": data_3d,
            "mean_3d": mean_3d,
            "std_3d": std_3d,
            "idx": idx,
        }

        if self.mode == "test":
            ret["data_2d_gt_flip"] = data_2d_gt_flip
            ret["data_2d_cpn_flip"] = data_2d_cpn_flip
            ret["data_3d_flip"] = data_3d_flip

        return ret
