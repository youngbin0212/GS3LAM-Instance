import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
import torch.nn.functional as F  # Edit
from skimage.measure import label  # Edit


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import open3d as o3d
# import keyboard
import torch
import imageio
import json

from src.utils.common_utils import seed_everything
from src.Decoder import SemanticDecoder
from visualizer.viser_utils import (
    load_camera_recon,
    load_scene_data,
    render_sam,
    rgbd2pcd,
    rgbd2pcd_sem,
    rgbd2pcd_sem_color,
    rgbd2pcd_sem_feature,
    make_lineset,
    save_screenshot,
)

os.makedirs('./output/frames', exist_ok=True)


# =======================
# Instance utilities
# =======================
# Edit

def generate_instance_ids(semantic_mask):
    instance_mask = np.zeros_like(semantic_mask, dtype=np.int32)
    for class_id in np.unique(semantic_mask):
        if class_id == 0:  # background 제외
            continue
        binary_mask = (semantic_mask == class_id)
        labeled = label(binary_mask)
        for i in range(1, labeled.max() + 1):
            instance_mask[labeled == i] = class_id * 1000 + i
    return instance_mask


# === mIoU 계산 함수 ===

def compute_miou(pred_logits, gt_semantic, num_classes):
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=0).cpu().numpy()
        gt = gt_semantic.cpu().numpy() if isinstance(gt_semantic, torch.Tensor) else gt_semantic

        ious = []
        for cls in range(num_classes):
            pred_mask = pred == cls
            gt_mask = gt == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union == 0:
                continue
            ious.append(intersection / union)

        return np.mean(ious) if ious else 0.0


# === Instance mIoU 계산 함수 ===

def compute_instance_miou(pred_instance, gt_instance):
    pred = pred_instance.cpu().numpy() if isinstance(pred_instance, torch.Tensor) else pred_instance
    gt = gt_instance.cpu().numpy() if isinstance(gt_instance, torch.Tensor) else gt_instance

    pred_ids = np.unique(pred)
    gt_ids = np.unique(gt)

    ious = []
    for gt_id in gt_ids:
        if gt_id == 0:  # background optional
            continue
        gt_mask = gt == gt_id
        best_iou = 0.0
        for pred_id in pred_ids:
            if pred_id == 0:
                continue
            pred_mask = pred == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)

    return np.mean(ious) if ious else 0.0


# =======================
# Helper: build decoder to match checkpoint 'inst_head' size
# =======================

def build_semantic_decoder(classifier_path, feature_dim=16, hidden_dim=256, device='cpu', force_inst_dim=None):
    """
    Instantiate SemanticDecoder so that its instance-embedding head matches the checkpoint.
    - If checkpoint has inst_head.weight (shape: [out_channels, in_channels, 1, 1]), use out_channels as inst_dim.
    - If `force_inst_dim` is given, build with that dim and load checkpoint with strict=False, dropping mismatched keys.
    """
    state = torch.load(classifier_path, map_location='cpu')

    # Infer inst_dim from checkpoint unless user forces a value
    inst_dim = None
    if force_inst_dim is None and 'inst_head.weight' in state:
        inst_dim = state['inst_head.weight'].shape[0]

    # Instantiate decoder (handle older signatures gracefully)
    try:
        if inst_dim is not None:
            model = SemanticDecoder(feature_dim, hidden_dim, inst_dim)
        elif force_inst_dim is not None:
            model = SemanticDecoder(feature_dim, hidden_dim, force_inst_dim)
        else:
            model = SemanticDecoder(feature_dim, hidden_dim)
    except TypeError:
        # Older SemanticDecoder(feature_dim, hidden_dim) without instance head
        model = SemanticDecoder(feature_dim, hidden_dim)

    # Load weights (allow partial if shapes differ)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] Loaded classifier with partial state: missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()

    used_inst_dim = inst_dim if inst_dim is not None else force_inst_dim
    if used_inst_dim is not None:
        print(f"[decoder] Instance embedding dim = {used_inst_dim} (from {'ckpt' if inst_dim is not None else 'force'})")
    else:
        print("[decoder] Instance head not found — running without instance embedding.")

    return model


def offine_recon(cfg, follow_cam=False):
    scene_path = os.path.join(cfg["logdir"], "params.npz")
    video_path = os.path.join(cfg["logdir"], cfg["render_mode"] + str("_it.mp4"))

    # Load Scene Data first
    w2c, k = load_camera_recon(cfg, scene_path)
    scene_data, all_w2cs = load_scene_data(scene_path, w2c, k)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=int(cfg['viz_w'] * cfg['view_scale']),
        height=int(cfg['viz_h'] * cfg['view_scale']),
        visible=True,
    )

    # Render one frame to probe device and/or initialize decoder
    im, depth, sem_feature = render_sam(w2c, k, scene_data, cfg)

    # === Build semantic decoder to MATCH CHECKPOINT inst_head size ===
    classifier_path = os.path.join(cfg["logdir"], "classifier.pth")
    dec_device = sem_feature.device if torch.is_tensor(sem_feature) else 'cpu'
    semantic_decoder = build_semantic_decoder(classifier_path, feature_dim=16, hidden_dim=256, device=dec_device)

    # Initialize point cloud for the chosen render mode
    if cfg['render_mode'] == 'color' or cfg['render_mode'] == 'depth':
        init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem':
        with torch.no_grad():
            logits, _ = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem(logits, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem_color':
        with torch.no_grad():
            logits, _ = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem_color(im, logits, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem_feature':
        init_pts, init_cols = rgbd2pcd_sem_feature(sem_feature, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'centers':
        init_pts = o3d.utility.Vector3dVector(
            scene_data['means3D'].contiguous().double().cpu().numpy()
        )
        init_cols = o3d.utility.Vector3dVector(
            scene_data['colors_precomp'].contiguous().double().cpu().numpy()
        )

    # Safety check
    if 'init_pts' in locals():
        print(f"#Points before visualization: {len(init_pts)}")
    else:
        print("[경고] init_pts가 정의되지 않았습니다. 시각화를 건너뜁니다.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    pcd = pcd.random_down_sample(0.05)
    print(f"#Points after downsampling: {len(pcd.points)}")

    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        red_colormap = LinearSegmentedColormap.from_list("red_colormap", [(0, 'red'), (1, 'red')])
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            if i_t % 50 != 0:
                continue
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(red_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')

        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(red_colormap((line_t * norm_factor / total_num_lines) + norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control
    view_control = vis.get_view_control()
    if os.path.exists(cfg['cam_json']):
        with open(cfg['cam_json'], 'r') as f:
            cam_params_dict = json.load(f)

        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = cam_params_dict['extrinsic']
        cparams.intrinsic.intrinsic_matrix = cam_params_dict['intrinsic']['intrinsic_matrix']
        cparams.intrinsic.height = cam_params_dict['intrinsic']['height']
        cparams.intrinsic.width = cam_params_dict['intrinsic']['width']

        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
    else:
        view_k = k * cfg['view_scale']
        view_k[2, 2] = 1
        cparams = o3d.camera.PinholeCameraParameters()
        if cfg['offset_first_viz_cam']:
            view_w2c = w2c
            view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
        else:
            view_w2c = w2c
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
        cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # =======================
    # Interactive Rendering (window visible + optional video)
    # =======================
    if cfg.get('save_video', False):
        frames = []

    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(
                scene_data['means3D'].contiguous().double().cpu().numpy()
            )
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'].contiguous().double().cpu().numpy()
            )
        else:
            im, depth, sem_feature = render_sam(w2c, k, scene_data, cfg)

            if cfg['render_mode'] == 'color' or cfg['render_mode'] == 'depth':
                pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem':
                with torch.no_grad():
                    logits, _ = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem(logits, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem_color':
                with torch.no_grad():
                    logits, _ = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem_color(im, logits, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem_feature':
                pts, cols = rgbd2pcd_sem_feature(sem_feature, depth, w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break

        vis.update_renderer()

        if cfg.get('save_video', False):
            img = vis.capture_screen_float_buffer(False)
            img_np = np.asarray(img)
            img_uint8 = (img_np * 255).astype('uint8')
            frames.append(img_uint8)

    if cfg.get('save_video', False):
        imageio.mimwrite(video_path, frames, fps=30)

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Path to experiment file")
    parser.add_argument(
        "--mode",
        type=str,
        default='color',
        help="['color, 'depth', 'centers', 'sem', 'sem_color', 'sem_feature']",
    )
    parser.add_argument("--plot_traj", type=bool, default=False, help="plot trajectory")
    parser.add_argument("--save_video", type=bool, default=False, help="save video")
    parser.add_argument("--cam_json", type=str, default='camera.json', help="camera params json")
    args = parser.parse_args()

    print(args.logdir)
    config_path = os.path.join(args.logdir, 'config.py')
    config = SourceFileLoader(os.path.basename(config_path), config_path).load_module()

    seed_everything(seed=config.seed)

    viz_cfg = config.config['viz']
    viz_cfg["render_mode"] = args.mode
    viz_cfg['save_video'] = args.save_video
    viz_cfg['cam_json'] = args.cam_json
    viz_cfg['logdir'] = args.logdir
    viz_cfg['visualize_cams'] = args.plot_traj

    # Visualize Final Reconstruction
    offine_recon(viz_cfg)
