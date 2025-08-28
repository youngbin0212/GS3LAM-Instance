import torch
import torch.nn.functional as F

from src.utils.gaussian_utils import transform_to_frame
from src.Render import transformed_params2rendervar
from src.utils.metric_utils import calc_ssim, l1_loss_v1            
from src.utils.contrastive_loss import SimpleContrastiveLoss

from gaussian_semantic_rasterization import GaussianRasterizer

def initialize_optimizer(params, lrs_dict, tracking):
    param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, 
             use_l1,ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, use_reg_loss=False,
             semantic_decoder=None,
             use_semantic_for_tracking=True,
             use_semantic_for_mapping=True,
             use_alpha_for_loss=False,
             alpha_thres=0.99,
             num_classes=256,
             instance_mask=None):   # 추가
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    
    # Rendering
    rendervar['means2D'].retain_grad()
    rendered_image, rendered_objects, radii, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(rendered_depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - rendered_depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask

    if tracking and use_alpha_for_loss:
        presence_alpha_mask = (rendered_alpha > alpha_thres)
        mask = mask & presence_alpha_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].mean()
    
    # RGB Loss
    if tracking and ignore_outlier_depth_loss:
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - rendered_image)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - rendered_image).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(rendered_image, curr_data['im']) + 0.2 * (1.0 - calc_ssim(rendered_image, curr_data['im']))
    
    gt_obj = curr_data["obj"].long()
    #logits = semantic_decoder(rendered_objects)
    #cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    sem_logits, inst_embed = semantic_decoder(rendered_objects)  # ← (C,H,W) 입력 가능
    cls_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    # ------------------------------------------------------------------
    # Instance compactness loss (옵션)
    # - instance_mask: (H, W), 0은 배경/무시, 동일 ID는 같은 인스턴스
    # - rendered_objects: (C, H, W) 를 픽셀 임베딩으로 사용
    # ------------------------------------------------------------------

    # Instance compactness loss (옵션) — mapping 단계에서만 실행
    #if (instance_mask is not None) and ('inst' in loss_weights) and (loss_weights['inst'] > 0):
    if mapping and (instance_mask is not None) and ('inst' in loss_weights) and (loss_weights['inst'] > 0):
        D, H, W = inst_embed.shape

        # 마스크 해상도 정렬
        if instance_mask.ndim == 2:
            m = instance_mask[None, None].float()
        else:
            m = instance_mask.float().view(1, 1, *instance_mask.shape[-2:])
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode='nearest')
        mid = m[0, 0].long().to(inst_embed.device)  # (H,W)

        # (선택) opacity로 신뢰도가 낮은 픽셀 무시
        if use_alpha_for_loss:
            pres = (rendered_alpha > alpha_thres)[0]       # (H,W) bool
            mid = torch.where(pres, mid, torch.zeros_like(mid))

        # 픽셀 임베딩 (HW, D)
        emb = inst_embed.permute(1, 2, 0).reshape(-1, D)   # (HW, D)
        ids = mid.reshape(-1)                              # (HW,)

        # -------- compactness (intra-instance) --------
        compact = emb.new_tensor(0.0); nvalid = 0
        for iid in torch.unique(ids):
            if iid <= 0:
                continue
            sel = (ids == iid)
            if sel.sum() < 5:
                continue
            v = emb[sel]                 # (N_i, D)
            c = v.mean(0, keepdim=True)  # (1, D)
            compact = compact + ((v - c) ** 2).mean()
            nvalid += 1

        if nvalid > 0:
            losses['inst'] = compact / nvalid
            # print(f"[loss.py debug] inst={float(losses['inst'])}")

        # ---- Contrastive instance 분리 손실 (선택, config에 inst_ctr > 0 일 때) ----
        if ('inst_ctr' in loss_weights) and (loss_weights['inst_ctr'] > 0):

            # 이미 계산된 ids:(HW,), emb:(HW,D) 재사용
            valid = ids > 0
            if valid.sum() > 50:  # 최소 샘플 수
                emb_valid = emb[valid]
                labels_valid = ids[valid]

                # 과다 시 샘플링 (메모리 방지)
                if emb_valid.shape[0] > 5000:
                    idx = torch.randperm(emb_valid.shape[0], device=emb_valid.device)[:5000]
                    emb_valid = emb_valid[idx]
                    labels_valid = labels_valid[idx]

                contrast_fn = SimpleContrastiveLoss(temperature=0.1)
                loss_ctr = contrast_fn(emb_valid, labels_valid)
                losses['inst_ctr'] = loss_ctr
        """ CHECKED: It worked!
        # ---- 디버그 출력 (50 프레임마다) ----
        if (iter_time_idx % 50 == 0):
            print("[loss debug] iter:", iter_time_idx,
                "inst:", float(losses.get('inst', 0.0)),
                "inst_ctr:", float(losses.get('inst_ctr', 0.0)))"""

    if tracking and use_semantic_for_tracking:
        ce_map = cls_criterion(sem_logits.unsqueeze(0), gt_obj.unsqueeze(0))[0]  # (H,W)

        if use_alpha_for_loss:
            # opacity(=rendered_alpha)로 신뢰도 가중
            w = rendered_alpha[0].clamp(0, 1)        # (H,W) 혹은 (1,H,W)->(H,W)
            loss_obj = (ce_map * w).sum() / (w > 0).sum().clamp_min(1)
        elif ignore_outlier_depth_loss:
            obj_mask = mask.detach().squeeze(0)      # (H,W)
            loss_obj = ce_map[obj_mask].sum()
        else:
            loss_obj = ce_map.sum()

        norm_denom = torch.log(torch.tensor(float(num_classes), device=sem_logits.device))
        losses['obj'] = loss_obj / norm_denom
    

    if mapping and use_semantic_for_mapping:
        ce_map = cls_criterion(sem_logits.unsqueeze(0), gt_obj.unsqueeze(0))[0]  # (H,W)
        if use_alpha_for_loss:
            w = rendered_alpha[0].clamp(0, 1)
            loss_obj = (ce_map * w).sum() / (w > 0).sum().clamp_min(1)
        else:
            loss_obj = ce_map.mean()

        norm_denom = torch.log(torch.tensor(float(num_classes), device=sem_logits.device))
        losses['obj'] = loss_obj / norm_denom


    # regularize Gaussians, scale, meters
    if mapping and use_reg_loss:
        scaling = torch.exp(params['log_scales'])
        mean_scale = scaling.mean()
        std_scale = scaling.std()
        # 1 sigma: 68.3%; 2 sigma 95.4%; 3 sigma 99.7%
        upper_limit = mean_scale + 2 * std_scale
        lower_limit = mean_scale - 2 * std_scale
        # regularize very big Gaussian
        if upper_limit < scaling.max():
            losses["big_gaussian_reg"] = torch.mean(scaling[torch.where(scaling > upper_limit)])
        else:
            losses["big_gaussian_reg"] = 0.0
        # regularize very small Gaussian
        if lower_limit > scaling.min():
            losses["small_gaussian_reg"] = torch.mean(-torch.log(scaling[torch.where(scaling < lower_limit)]))
        else:
            losses["small_gaussian_reg"] = 0.0

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radii > 0
    variables['max_2D_radius'][seen] = torch.max(radii[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses