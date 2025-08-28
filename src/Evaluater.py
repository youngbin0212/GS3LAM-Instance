import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.decomposition import PCA

from src.Render import get_rasterizationSettings, transformed_params2rendervar
from src.utils.gaussian_utils import build_rotation, transform_to_frame
from src.utils.metric_utils import calc_psnr, evaluate_ate

from gaussian_semantic_rasterization import GaussianRasterizer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
loss_fn_alex.eval()  # 안전: eval 고정

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb

def feature_to_rgb(features):
    # features: (C, H, W), e.g., C=16
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())
    pca_result = pca_result.reshape(H, W, 3)
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min() + 1e-6)
    rgb_array = pca_normalized.astype('uint8')
    return rgb_array

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def _call_classifier(classifier, rendered_objects):
    """
    classifier가 (logits) 또는 (logits, inst_embed) 둘 다를 반환해도 안전하게 받기 위한 유틸.
    logits: (num_classes, H, W) 가정
    inst_embed: (E, H, W) 가정 (옵션)
    """
    out = classifier(rendered_objects)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        logits, inst_embed = out[0], out[1]
    else:
        logits, inst_embed = out, None
    return logits, inst_embed

def eval(dataset, final_params, num_frames, eval_dir,
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False,
         eval_every=1, save_frames=False, use_semantic=False, classifier=None,
         #추가 옵션: instance 분할 (기본 OFF → previous와 동일 속도/동작)
         do_instance=True,                 # 인스턴스 후처리 on/off
         inst_post_stride=1,                # 인스턴스 후처리 프레임 간격(1=매 프레임, previous와 동일 동작을 위해 기본 1)
         save_inst_products=True            # inst npy 저장 여부
         ):
    print("Evaluating Final Parameters ...")
    psnr_list, rmse_list, l1_list = [], [], []
    fps_list, lpips_list, ssim_list = [], [], []

    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    #previous의 동작을 보존: 내부에서 save_frames=True로 강제
    save_frames = True

    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        if use_semantic:
            object_dir = os.path.join(eval_dir, "object_mask")
            os.makedirs(object_dir, exist_ok=True)
            render_object_dir = os.path.join(eval_dir, "rendered_object")
            os.makedirs(render_object_dir, exist_ok=True)
            objects_feature16_dir = os.path.join(eval_dir, "objects_feature16")
            os.makedirs(objects_feature16_dir, exist_ok=True)
            # mIoU
            gt_mask_array_path = os.path.join(eval_dir, "gt_mask_array")
            os.makedirs(gt_mask_array_path, exist_ok=True)
            pred_mask_array_path = os.path.join(eval_dir, "pred_mask_array")
            os.makedirs(pred_mask_array_path, exist_ok=True)
            # instance 결과 저장 폴더 (옵션)
            inst_mask_array_path = os.path.join(eval_dir, "pred_inst_array")
            if do_instance and save_inst_products:
                os.makedirs(inst_mask_array_path, exist_ok=True)

    # cal miou
    gt_mask_list, pred_mask_list = [], []
    gt_w2c_list = []
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        sample = dataset[time_idx]
        if len(sample) == 7:
            color, depth, intrinsics, pose, gt_objects, inst_mask, _ = sample
        elif len(sample) == 6:
            color, depth, intrinsics, pose, gt_objects, inst_mask = sample
        elif len(sample) == 5:
            color, depth, intrinsics, pose, gt_objects = sample
            inst_mask = None
        else:
            color, depth, intrinsics, pose = sample
            gt_objects, inst_mask = None, None

        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)

        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = get_rasterizationSettings(color.shape[2], color.shape[1],
                                            intrinsics.cpu().numpy(),
                                            first_frame_w2c.detach().cpu().numpy())

        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(final_params, time_idx,
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Define current frame data
        if use_semantic:
            curr_data = {'cam': cam, 'im': color, 'depth': depth, "obj": gt_objects,
                         'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}
        else:
            curr_data = {'cam': cam, 'im': color, 'depth': depth,
                         'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        # Render
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        iter_start.record()
        im, rendered_objects, radius, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)
        iter_end.record()

        torch.cuda.synchronize()
        iter_time = iter_start.elapsed_time(iter_end) / 1000.0
        fps_list.append(1.0 / iter_time)

        # Metrics (PSNR/SSIM/LPIPS) — previous와 동일
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rendered_depth.detach()
        weighted_im = im * valid_depth_mask
        weighted_gt_im = curr_data['im'] * valid_depth_mask

        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(),
                       weighted_gt_im.unsqueeze(0).cpu(),
                       data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(
            torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
            torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)
        ).item()

        # Semantic + (옵션) Instance
        if use_semantic:
            logits, inst_embed = _call_classifier(classifier, rendered_objects)
            pred_obj = torch.argmax(logits, dim=0)

            # 저장용 배열 (mIoU 계산 등)
            gt_mask_array = gt_objects.cpu().numpy().astype(np.uint8)
            pred_mask_array = pred_obj.cpu().numpy().astype(np.uint8)
            np.save(os.path.join(gt_mask_array_path, f"gt_{time_idx:04d}.npy"), gt_mask_array)
            np.save(os.path.join(pred_mask_array_path, f"pred_{time_idx:04d}.npy"), pred_mask_array)

            # 시각화
            pred_obj_mask = visualize_obj(pred_mask_array)
            gt_rgb_mask = visualize_obj(gt_mask_array)
            rgb_mask = feature_to_rgb(rendered_objects)

            # === Instance post-processing (옵션) ===
            if do_instance and (inst_embed is not None) and (time_idx % inst_post_stride == 0):
                from src.utils.instance_post import segment_instances_from_embeddings
                pred_inst = segment_instances_from_embeddings(
                    logits, inst_embed, alpha=rendered_alpha[0], sim_thr=0.95, min_area=20
                )
                if save_inst_products:
                    pred_inst_np = pred_inst.cpu().numpy().astype(np.int32)
                    np.save(os.path.join(inst_mask_array_path, f"inst_{time_idx:04d}.npy"), pred_inst_np)

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Depth RMSE / L1 — previous와 동일
        if mapping_iters == 0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt(((rendered_depth - curr_data['depth']) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rendered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rendered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rendered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Rendered
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin, vmax = 0, 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, f"gs_{time_idx:04d}.png"),
                        cv2.cvtColor(viz_render_im * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, f"gs_{time_idx:04d}.png"), depth_colormap)

            # GT
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, f"gt_{time_idx:04d}.png"),
                        cv2.cvtColor(viz_gt_im * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, f"gt_{time_idx:04d}.png"), depth_colormap)

            if use_semantic:
                cv2.imwrite(os.path.join(object_dir, f"gt_{time_idx:04d}.png"), gt_rgb_mask)
                cv2.imwrite(os.path.join(render_object_dir, f"gs_{time_idx:04d}.png"), pred_obj_mask)
                cv2.imwrite(os.path.join(objects_feature16_dir, f"{time_idx:04d}.png"), rgb_mask)

    avg_metric = []
    avg_metric.append("Rendering FPS: {:.5f}".format(sum(fps_list) / len(fps_list)))

    try:
        # ATE RMSE — previous와 동일
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = [latest_est_w2c]
        valid_gt_w2c_list = [gt_w2c_list[0]]
        for idx in range(1, num_frames):
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        avg_metric.append("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse, "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
        avg_metric.append('Failed to evaluate trajectory with alignment.')

    # Averages — previous와 동일
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()

    print("Average PSNR: {:.3f}".format(avg_psnr))
    print("Average Depth RMSE: {:.3f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.3f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    avg_metric.append("Average PSNR: {:.3f}".format(avg_psnr))
    avg_metric.append("Average Depth RMSE: {:.3f} cm".format(avg_rmse*100))
    avg_metric.append("Average Depth L1: {:.3f} cm".format(avg_l1*100))
    avg_metric.append("Average MS-SSIM: {:.3f}".format(avg_ssim))
    avg_metric.append("Average LPIPS: {:.3f}".format(avg_lpips))
    avg_metric = np.array(avg_metric, dtype='str')
    np.savetxt(os.path.join(eval_dir, "avg_metric.txt"), avg_metric, fmt='%s')

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Plot PSNR & L1 as line plots — previous와 동일
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(
        avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()
