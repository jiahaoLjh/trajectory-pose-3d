import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):

    def __init__(self, config):
        super(PoseNet, self).__init__()

        self.config = config
        # zero root joint is not regressed
        self.n_joints = config["n_joints"] - 1
        self.n_bases = config["n_bases"]
        self.n_frames = config["n_frames"]

        self.conv_feats = nn.ModuleList([
            nn.Conv1d((self.n_joints + 1) * 2, 256, 1, stride=1, padding=0),
            nn.Conv1d((self.n_joints + 1) * 2 + 256, 256, 1, stride=1, padding=0),
            nn.Conv1d((self.n_joints + 1) * 2 + 256 * 2, 256, 1, stride=1, padding=0),
            nn.Conv1d((self.n_joints + 1) * 2 + 256 * 3, 256, 1, stride=1, padding=0),
            nn.Conv1d((self.n_joints + 1) * 2 + 256 * 4, 256, 1, stride=1, padding=0),
        ])

        self.bn_feats = nn.ModuleList([
            nn.BatchNorm1d(256) for _ in range(len(self.conv_feats))
        ])

        self.dense_mlps = nn.ModuleList([
            nn.Linear(((self.n_joints + 1) * 2 + 256 * 5) * self.n_bases, 1024),
            nn.Linear(((self.n_joints + 1) * 2 + 256 * 5) * self.n_bases + 1024, 1024),
            nn.Linear(((self.n_joints + 1) * 2 + 256 * 5) * self.n_bases + 1024 * 2, 1024),
            nn.Linear(((self.n_joints + 1) * 2 + 256 * 5) * self.n_bases + 1024 * 3, 1024),
        ])

        self.bn_mlps = nn.ModuleList([
            nn.BatchNorm1d(1024) for _ in range(len(self.dense_mlps))
        ])

        self.dense_pred = nn.Linear(((self.n_joints + 1) * 2 + 256 * 5) * self.n_bases + 1024 * 4, self.n_joints * 3 * self.n_bases)

    def forward(self, w_2d, bases):
        """

        :param w_2d: (B, Jx2, F)
        :param bases: (B, K, F)
        :return: (B, Jx3, K)
        """
        out = w_2d

        feats = [out]

        for i in range(len(self.conv_feats)):
            out = torch.cat(feats, 1)
            out = self.conv_feats[i](out)
            out = self.bn_feats[i](out)
            out = F.relu(out)
            out = F.dropout(out, 0.5, self.training)
            feats.append(out)

        feats = F.avg_pool1d(torch.cat(feats, 1), 5, stride=1, padding=2)

        transformed_feats = torch.matmul(feats, torch.transpose(bases, 1, 2)) / self.n_frames * 2
        transformed_feats = transformed_feats.view(transformed_feats.shape[0], -1)  # (B, C, K)

        fused_layers = [transformed_feats]

        for i in range(len(self.dense_mlps)):
            out = torch.cat(fused_layers, 1)
            out = self.dense_mlps[i](out)
            out = self.bn_mlps[i](out)
            out = F.relu(out)
            out = F.dropout(out, 0.25, self.training)
            fused_layers.append(out)

        coeff = self.dense_pred(torch.cat(fused_layers, 1))
        data_3d = coeff.view(-1, self.n_joints * 3, self.n_bases)  # (B, Jx3, K)

        return data_3d

    def build_loss_training(self, coeff, bases, pose_3d, mean_3d, std_3d):
        """
        Build loss for the training stage.
        """

        B = coeff.shape[0]

        pred_3d = torch.matmul(coeff, bases)  # (B, Jx3, F)

        # un-normalize 3d
        pred_3d = pred_3d * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)
        gt_3d = pose_3d * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)

        total_loss = torch.mean(torch.abs(pred_3d - gt_3d))

        return total_loss

    def build_loss_test(self, coeff, bases, pose_3d, mean_3d, std_3d):
        """
        Build loss for the test stage. Original and flipped videos are both taken as data augmentation.
        Estimated 3d poses are returned.
        """

        B = coeff[0].shape[0]

        pred_3d = torch.matmul(coeff[0], bases)  # (B, Jx3, F)
        pred_3d_flip = torch.matmul(coeff[1], bases)  # (B, Jx3, F)

        # un-normalize 3d
        pred_3d = pred_3d * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)
        pred_3d_flip = pred_3d_flip * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)
        gt_3d = pose_3d[0] * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)
        gt_3d_flip = pose_3d[1] * std_3d.view(B, self.n_joints * 3, 1) + mean_3d.view(B, self.n_joints * 3, 1)

        total_loss = (torch.mean(torch.abs(pred_3d - gt_3d)) + torch.mean(torch.abs(pred_3d_flip - gt_3d_flip))) / 2

        # prediction
        np_pred_3d = pred_3d.cpu().data.numpy()
        np_pred_3d = np.reshape(np_pred_3d, [B, self.n_joints, 3, self.n_frames])
        np_pred_3d_flip = pred_3d_flip.cpu().data.numpy()
        np_pred_3d_flip = np.reshape(np_pred_3d_flip, [B, self.n_joints, 3, self.n_frames])
        np_gt_3d = gt_3d.cpu().data.numpy()
        np_gt_3d = np.reshape(np_gt_3d, [B, self.n_joints, 3, self.n_frames])

        assert self.n_joints == 16, self.n_joints
        left_right_symmetry = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12])

        np_pred_3d_flip[:, :, 0, :] *= -1
        np_pred_3d_flip = np_pred_3d_flip[:, left_right_symmetry, :, :]

        np_pred_3d_avg = (np_pred_3d + np_pred_3d_flip) / 2

        return total_loss, (np_pred_3d_avg, np_gt_3d)
