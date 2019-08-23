import tqdm
import numpy as np
from utils import compute_similarity_transform


def h36m_evaluate(preds_3d, gts_3d, indices, test_dataset, config):
    """
    Evaluate on Human3.6M dataset. Action-wise and overall errors are measured.
    """
    logger = config["logger"]
    logger.info("Evaluating on videos...")

    all_frames_preds = []
    all_frames_gts = []
    all_frames_indices = []

    current_video_id = -1
    current_frame_id = -1

    n_sample = gts_3d.shape[0]

    video_start_indices = []

    index_to_action = {}
    action_preds = {}
    action_gts = {}
    for idx, k in enumerate(sorted(test_dataset.data_3d)):
        index_to_action[idx] = k[1]
        if k[1] not in action_preds:
            action_preds[k[1]] = []
            action_gts[k[1]] = []

    for i in tqdm.tqdm(range(n_sample)):
        if indices[i] != current_video_id:
            # start in a new video
            current_video_id = indices[i]
            current_frame_id = 0
            video_start_indices.append(len(all_frames_preds))

        for t in range(0 if current_frame_id == 0 else config["n_frames"] - config["window_slide"], config["n_frames"]):
            frm = []
            for j in range(t // config["window_slide"] + 1):
                if i + j >= n_sample or indices[i + j] != current_video_id:
                    break
                # collect estimations from multiple samples which have overlapping at the current frame t
                frm.append(preds_3d[i + j, :, :, t - j * config["window_slide"]])
            frm = np.array(frm)
            frm = np.mean(frm, axis=0)
            all_frames_preds.append(frm)
            all_frames_gts.append(gts_3d[i, :, :, t])
            all_frames_indices.append(indices[i])

            current_frame_id += 1

    all_frames_preds = np.array(all_frames_preds)
    all_frames_gts = np.array(all_frames_gts)
    all_frames_indices = np.array(all_frames_indices)

    # add back the root joints
    all_frames_preds = np.concatenate([np.zeros([all_frames_preds.shape[0], 1, 3]), all_frames_preds], axis=1)
    all_frames_gts = np.concatenate([np.zeros([all_frames_gts.shape[0], 1, 3]), all_frames_gts], axis=1)

    for idx, start in enumerate(video_start_indices):
        if idx + 1 == len(video_start_indices):
            cp = all_frames_preds[start:]
            cg = all_frames_gts[start:]
            ci = all_frames_indices[start:]
        else:
            cp = all_frames_preds[start:video_start_indices[idx+1]]
            cg = all_frames_gts[start:video_start_indices[idx+1]]
            ci = all_frames_indices[start:video_start_indices[idx+1]]
        assert ci[0] == ci[-1]
        ci = ci[0]
        action_preds[index_to_action[ci]].append(cp)
        action_gts[index_to_action[ci]].append(cg)

    allp = []
    allg = []
    for act in sorted(action_preds):
        lp = np.concatenate(action_preds[act], axis=0)
        lg = np.concatenate(action_gts[act], axis=0)
        allp.append(lp)
        allg.append(lg)

        mpjpe, pampjpe = error(lp, lg, config)
        print("{:15s} {:>6d} frames, MPJPE = {:.3f}, PAMPJPE = {:.3f}".format(act, lp.shape[0], mpjpe, pampjpe))

    allp = np.concatenate(allp, axis=0)
    allg = np.concatenate(allg, axis=0)
    mpjpe, pampjpe = error(allp, allg, config)
    print("{:15s} {:>6d} frames, MPJPE = {:.3f}, PAMPJPE = {:.3f}".format("All", allp.shape[0], mpjpe, pampjpe))


def error(preds, gts, config):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    """
    N = preds.shape[0]

    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    pampjpe = np.zeros([N, config["n_joints"]])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe
