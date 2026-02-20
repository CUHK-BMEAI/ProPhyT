"""
是ctp测试代码，这里只看dense_prompt模式即可
支持三种融合模式的测试：cross_attention / dense_prompt / sparse_prompt
"""

from segment_anything import sam_model_registry
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import json
import re
import csv
from collections import defaultdict
from typing import Optional, List, Tuple
from pathlib import Path
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
import pandas as pd


from DataLoader import TestingDataset
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from metrics import SegMetrics, dice_by_category
from cpal_sam_modules import build_cpal_sam_model


def parse_args():
    parser = argparse.ArgumentParser(description='CM-CPAL SAM测试 v2')


    parser.add_argument("--work_dir", type=str, default="workdir", help="工作目录")
    parser.add_argument("--run_name", type=str, default="cpal_sam_v2", help="运行名称")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument("--data_path", type=str, default="data_penumbra_noblank_withvalid",
                       help="测试数据路径")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="评估指标")


    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM模型类型")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="训练好的CM-CPAL SAM checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                       help="SAM预训练checkpoint（测试时不需要）")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="使用adapter")


    parser.add_argument("--stage1b_ckpt", type=str, required=True,
                       help="Stage 1b checkpoint路径")
    parser.add_argument("--prototype_bank", type=str, required=True,
                       help="CTP原型库路径")
    parser.add_argument("--fusion_mode", type=str, default="cross_attention",
                       choices=['cross_attention', 'dense_prompt', 'sparse_prompt'],
                       help="融合模式")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--top_k", type=int, default=5, help="检索的原型数量")
    parser.add_argument("--temperature", type=float, default=0.1, help="检索温度参数")
    parser.add_argument("--random_prototype", action='store_true',
                       help="消融实验：随机选择原型而非基于相似度检索")
    parser.add_argument("--hard_weight", action='store_true',
                       help="消融实验：使用hard weight（one-hot）而非soft weight")

    # 测试参数
    parser.add_argument("--boxes_prompt", type=str, default="True", help="使用boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="点数")
    parser.add_argument("--iter_point", type=int, default=1, help="点迭代次数")
    parser.add_argument("--multimask", type=str, default="True", help="输出多个mask")
    parser.add_argument("--prompt_path", type=str, default=None, help="固定prompt路径")
    parser.add_argument("--save_pred", type=str, default="False", help="保存预测结果")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path (default: results_cpal_sam.csv)")
    parser.add_argument("--nssd_tau", type=float, default=2.0, help="NSD tolerance")

    args = parser.parse_args()


    args.boxes_prompt = args.boxes_prompt.lower() in ['true', '1', 'yes']
    args.multimask = args.multimask.lower() in ['true', '1', 'yes']
    args.save_pred = args.save_pred.lower() in ['true', '1', 'yes']

    if args.iter_point > 1:
        args.point_num = 1

    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key in ['image', 'label']:
                device_input[key] = value.float().to(device)
            elif type(value) in [list, torch.Size]:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def postprocess_masks(low_res_masks, image_size, original_size):
    """后处理masks"""
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc')
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    
    return masks, pad


def prompt_and_decoder(args, batched_input, cpal_sam_model, image_embeddings,
                       retrieved_prototypes=None, weights=None):
    """
    Prompt编码和Mask解码
    """
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    
    with torch.no_grad():
        if args.fusion_mode in ['dense_prompt', 'sparse_prompt']:
            # 使用prompt替换模式
            if args.fusion_mode == 'dense_prompt':
                sparse_embeddings, _ = cpal_sam_model.sam_model.prompt_encoder(
                    points=points,
                    boxes=batched_input.get("boxes", None),
                    masks=None
                )
                dense_embeddings = cpal_sam_model.prompt_replacer(
                    retrieved_prototypes, weights
                )
            else:  # sparse_prompt
                _, dense_embeddings = cpal_sam_model.sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=batched_input.get("mask_inputs", None)
                )
                sparse_embeddings = cpal_sam_model.prompt_replacer(
                    retrieved_prototypes, weights
                )
        else:
            # cross_attention模式，使用原先标准prompt encoder，也就是self-prompting
            sparse_embeddings, dense_embeddings = cpal_sam_model.sam_model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
        
        low_res_masks, iou_predictions = cpal_sam_model.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=cpal_sam_model.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    
    masks = F.interpolate(
        low_res_masks,
        (args.image_size, args.image_size),
        mode="bilinear",
        align_corners=False
    )
    
    return masks, low_res_masks, iou_predictions


def extract_subject_id_from_name(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    m = re.match(r'^(.*?)(?:_[0-9]+)$', base)
    return m.group(1) if m else base


def dice_3d_from_volumes(pr_vol: np.ndarray, gt_vol: np.ndarray) -> float:
    pr_sum = pr_vol.sum()
    gt_sum = gt_vol.sum()
    if pr_sum > 0 and gt_sum > 0:
        inter = np.logical_and(pr_vol > 0, gt_vol > 0).sum()
        return (2.0 * inter) / (pr_sum + gt_sum + 1e-7)
    elif pr_sum > 0 and gt_sum == 0:
        return 0.0
    elif pr_sum == 0 and gt_sum > 0:
        return 0.0
    else:
        return 1.0


def _get_surface_points(mask):
    """Extract surface voxels from a binary mask using erosion."""
    if mask.sum() == 0:
        return np.array([]).reshape(0, mask.ndim)
    eroded = binary_erosion(mask, iterations=1)
    surface = np.logical_and(mask, ~eroded)
    return np.argwhere(surface)


def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


def compute_precision(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0:
        return 0.0
    return np.logical_and(pred, gt).sum() / (pred.sum() + 1e-8)


def compute_recall(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if gt.sum() == 0:
        return 0.0
    return np.logical_and(pred, gt).sum() / (gt.sum() + 1e-8)


def compute_hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    gt_dist_map = distance_transform_edt(~gt)
    pred_dist_map = distance_transform_edt(~pred)
    pred_to_gt = gt_dist_map[pred]
    gt_to_pred = pred_dist_map[gt]
    return max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95))


def compute_nssd(pred, gt, tau=2.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    pred_surface = _get_surface_points(pred)
    gt_surface = _get_surface_points(gt)
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 0.0
    gt_dist_map = distance_transform_edt(~gt)
    pred_dist_map = distance_transform_edt(~pred)
    pred_to_gt = gt_dist_map[tuple(pred_surface.T)]
    gt_to_pred = pred_dist_map[tuple(gt_surface.T)]
    frac_pred = (pred_to_gt <= tau).sum() / len(pred_to_gt)
    frac_gt = (gt_to_pred <= tau).sum() / len(gt_to_pred)
    return (frac_pred + frac_gt) / 2.0


def main(args):
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('*'*100)
    
    sam_model = sam_model_registry[args.model_type](args).to(args.device)

    cpal_sam_model = build_cpal_sam_model(
        sam_model=sam_model,
        stage1b_ckpt_path=args.stage1b_ckpt,
        prototype_bank_path=args.prototype_bank,
        fusion_mode=args.fusion_mode,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        temperature=args.temperature,
        freeze_ncct=True,
        random_selection=args.random_prototype,
        hard_weight=args.hard_weight,
    ).to(args.device)
    
 
    print(f"*******load checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    cpal_sam_model.load_state_dict(state_dict, strict=False)

    

    cpal_sam_model.eval()
    
    criterion = FocalDiceloss_IoULoss()
    
    test_dataset = TestingDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        mode='test',
        requires_name=True,
        point_num=args.point_num,
        return_ori_mask=True,
        prompt_path=args.prompt_path
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    
    test_pbar = tqdm(test_loader)
    l = len(test_loader)
    
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}
    
    # 分类别统计
    all_core_dice = []
    all_penumbra_dice = []
    core_count_total = 0
    penumbra_count_total = 0
    core_empty_total = 0
    penumbra_empty_total = 0
    core_nonempty_dice_all = []
    penumbra_nonempty_dice_all = []
    core_empty_gt_pred_empty_total = 0
    penumbra_empty_gt_pred_empty_total = 0
    core_pred_nonempty_total = 0
    penumbra_pred_nonempty_total = 0
    
  
    subj_cat_pred_slices = defaultdict(list)
    subj_cat_gt_slices = defaultdict(list)
    
   
    all_retrieved_indices = []
    all_retrieval_weights = []

    
    all_slice_results = []
    metrics_list = ["Dice", "IoU", "HD95", "NSD", "F1", "Precision", "Recall"]
    
    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }
        
        with torch.no_grad():
            ncct_features, ncct_latent = cpal_sam_model.ncct_feature_extractor(
                cpal_sam_model._prepare_ncct_input(batched_input["image"])
            )#提取ctp proto
            retrieved_prototypes, weights, indices = cpal_sam_model.prototype_retriever(ncct_latent)
            
        
            all_retrieved_indices.append(indices.cpu().numpy())
            all_retrieval_weights.append(weights.cpu().numpy())
            
            # SAM image encoder
            image_embeddings = cpal_sam_model.sam_model.image_encoder(batched_input["image"])
            
            # 这里是消融的cross_attention
            if args.fusion_mode == 'cross_attention':
                image_embeddings = cpal_sam_model.cross_attention_fusion(
                    sam_features=image_embeddings,
                    ctp_prototypes=retrieved_prototypes,
                    weights=weights
                )
        
        
        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, cpal_sam_model, image_embeddings,
                retrieved_prototypes, weights
            )
            points_show = None
        else:
            save_path = os.path.join(
                f"{args.work_dir}", args.run_name,
                f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt"
            )
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
            
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, cpal_sam_model, image_embeddings,
                    retrieved_prototypes, weights
                )
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords, dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
            
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))
        
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad,
                      batched_input.get("boxes", None), points_show)
        
        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())
        
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
        
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
        
       
        mask_paths = batched_input.get("mask_path", None)
        if mask_paths is None:
            mask_paths = [batched_input['name'][0]]
        if isinstance(mask_paths, str):
            mask_paths = [mask_paths]
        
        dice_cat = dice_by_category(masks, ori_labels, mask_paths=mask_paths)
        if len(dice_cat['core_dice']) > 0:
            all_core_dice.extend(dice_cat['core_dice'].tolist())
        if len(dice_cat['penumbra_dice']) > 0:
            all_penumbra_dice.extend(dice_cat['penumbra_dice'].tolist())
        core_count_total += dice_cat['core_count']
        penumbra_count_total += dice_cat['penumbra_count']
        core_empty_total += dice_cat.get('core_empty_count', 0)
        penumbra_empty_total += dice_cat.get('penumbra_empty_count', 0)
        
        if len(dice_cat.get('core_nonempty_dice', [])) > 0:
            core_nonempty_dice_all.extend(
                dice_cat['core_nonempty_dice'].tolist() 
                if hasattr(dice_cat['core_nonempty_dice'], "tolist") 
                else list(dice_cat['core_nonempty_dice'])
            )
        if len(dice_cat.get('penumbra_nonempty_dice', [])) > 0:
            penumbra_nonempty_dice_all.extend(
                dice_cat['penumbra_nonempty_dice'].tolist()
                if hasattr(dice_cat['penumbra_nonempty_dice'], "tolist")
                else list(dice_cat['penumbra_nonempty_dice'])
            )
        core_empty_gt_pred_empty_total += dice_cat.get('core_empty_gt_pred_empty_count', 0)
        penumbra_empty_gt_pred_empty_total += dice_cat.get('penumbra_empty_gt_pred_empty_count', 0)
        core_pred_nonempty_total += dice_cat.get('core_pred_nonempty_count', 0)
        penumbra_pred_nonempty_total += dice_cat.get('penumbra_pred_nonempty_count', 0)
        
       
        img_name_lower = img_name.lower()
        category = 'core' if ('core' in img_name_lower and 'penumbra' not in img_name_lower) else 'penumbra'
        subject_id = extract_subject_id_from_name(img_name)
        
        pr_bin = (masks > 0.5).detach().cpu().numpy().astype(np.uint8)[0, 0]
        gt_bin = (ori_labels > 0.5).detach().cpu().numpy().astype(np.uint8)[0, 0]
        subj_cat_pred_slices[(subject_id, category)].append(pr_bin)
        subj_cat_gt_slices[(subject_id, category)].append(gt_bin)

       
        gt_nonempty = gt_bin.sum() > 0
        is_nonempty = (pr_bin.sum() > 0) or gt_nonempty
        d = compute_dice(pr_bin, gt_bin)
        iou_val = compute_iou(pr_bin, gt_bin)
        hd95_val = compute_hd95(pr_bin, gt_bin)
        nsd_val = compute_nssd(pr_bin, gt_bin, tau=args.nssd_tau)
        prec_val = compute_precision(pr_bin, gt_bin)
        rec_val = compute_recall(pr_bin, gt_bin)
        all_slice_results.append({
            "patient": subject_id, "slice_name": img_name, "category": category,
            "nonempty": is_nonempty, "gt_nonempty": gt_nonempty,
            "Dice": d, "IoU": iou_val, "HD95": hd95_val, "NSD": nsd_val, "F1": d,
            "Precision": prec_val, "Recall": rec_val,
        })
    

    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {
        args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i])
        for i in range(len(test_iter_metrics))
    }
    
    average_loss = np.mean(test_loss)
    
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir, f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    
 
    print(f"\ntest loss: {average_loss:.4f}, indictaor: {test_metrics}")
    
    # 分类别Dice统计
    core_dice_mean = np.mean(all_core_dice) if len(all_core_dice) > 0 else 0.0
    penumbra_dice_mean = np.mean(all_penumbra_dice) if len(all_penumbra_dice) > 0 else 0.0
    
    print("\n" + "="*80)
    print("="*80)
    print(f"Core mask:")
    print(f"  - 总数量: {core_count_total}")
    print(f"  - 空mask数量: {core_empty_total}")
    print(f"  - Dice均值: {core_dice_mean:.4f}")
    if len(all_core_dice) > 0:
        print(f"  - Dice标准差: {np.std(all_core_dice):.4f}")
        print(f"  - Dice最小值: {np.min(all_core_dice):.4f}")
        print(f"  - Dice最大值: {np.max(all_core_dice):.4f}")
    
    core_nonempty_mean = np.mean(core_nonempty_dice_all) if len(core_nonempty_dice_all) > 0 else 0.0
    core_empty_acc = (core_empty_gt_pred_empty_total / core_empty_total) if core_empty_total > 0 else 0.0
    core_pred_nonempty_rate = (core_pred_nonempty_total / core_count_total) if core_count_total > 0 else 0.0
    core_nonempty_std = np.std(core_nonempty_dice_all) if len(core_nonempty_dice_all) > 0 else 0.0
    print(f"  - 非空样本Dice均值: {core_nonempty_mean:.4f}")
    print(f"  - 非空样本Dice标准差: {core_nonempty_std:.4f}")
    print(f"  - 空GT预测为空准确率: {core_empty_acc:.4f}")
    print(f"  - 预测为非空比例: {core_pred_nonempty_rate:.4f}")
    
    print(f"\nPenumbra mask:")
    print(f"  - 总数量: {penumbra_count_total}")
    print(f"  - 空mask数量: {penumbra_empty_total}")
    print(f"  - Dice均值: {penumbra_dice_mean:.4f}")
    if len(all_penumbra_dice) > 0:
        print(f"  - Dice标准差: {np.std(all_penumbra_dice):.4f}")
        print(f"  - Dice最小值: {np.min(all_penumbra_dice):.4f}")
        print(f"  - Dice最大值: {np.max(all_penumbra_dice):.4f}")
    
    penumbra_nonempty_mean = np.mean(penumbra_nonempty_dice_all) if len(penumbra_nonempty_dice_all) > 0 else 0.0
    penumbra_nonempty_std = np.std(penumbra_nonempty_dice_all) if len(penumbra_nonempty_dice_all) > 0 else 0.0
    penumbra_empty_acc = (penumbra_empty_gt_pred_empty_total / penumbra_empty_total) if penumbra_empty_total > 0 else 0.0
    penumbra_pred_nonempty_rate = (penumbra_pred_nonempty_total / penumbra_count_total) if penumbra_count_total > 0 else 0.0
    print(f"  - 非空样本Dice均值: {penumbra_nonempty_mean:.4f}")
    print(f"  - 非空样本Dice标准差: {penumbra_nonempty_std:.4f}")
    print(f"  - 空GT预测为空准确率: {penumbra_empty_acc:.4f}")
    print(f"  - 预测为非空比例: {penumbra_pred_nonempty_rate:.4f}")
    print("="*80)
    
    # 3D Dice统计
    core_3d_dice_list = []
    penumbra_3d_dice_list = []
    
    for (subject_id, category), pr_slices in subj_cat_pred_slices.items():
        gt_slices = subj_cat_gt_slices.get((subject_id, category), [])
        if len(pr_slices) == 0 or len(gt_slices) == 0:
            continue
        
        pr_vol = np.stack(pr_slices, axis=0).astype(np.uint8)
        gt_vol = np.stack(gt_slices, axis=0).astype(np.uint8)
        d3 = dice_3d_from_volumes(pr_vol, gt_vol)
        
        if category == 'core':
            core_3d_dice_list.append(d3)
        else:
            penumbra_3d_dice_list.append(d3)
    
    print("\n" + "="*80)
    print("3D Dice（按受试者聚合）")
    print("="*80)
    print("Core 3D Dice:")
    print(f"  - 受试者数量: {len(core_3d_dice_list)}")
    if len(core_3d_dice_list) > 0:
        print(f"  - 均值: {np.mean(core_3d_dice_list):.4f}")
        print(f"  - 标准差: {np.std(core_3d_dice_list):.4f}")
        print(f"  - 最小值: {np.min(core_3d_dice_list):.4f}")
        print(f"  - 最大值: {np.max(core_3d_dice_list):.4f}")
    
    print("\nPenumbra 3D Dice:")
    print(f"  - 受试者数量: {len(penumbra_3d_dice_list)}")
    if len(penumbra_3d_dice_list) > 0:
        print(f"  - 均值: {np.mean(penumbra_3d_dice_list):.4f}")
        print(f"  - 标准差: {np.std(penumbra_3d_dice_list):.4f}")
        print(f"  - 最小值: {np.min(penumbra_3d_dice_list):.4f}")
        print(f"  - 最大值: {np.max(penumbra_3d_dice_list):.4f}")
    print("="*80)
    
    # 原型检索统计
    all_retrieved_indices = np.concatenate(all_retrieved_indices, axis=0)
    all_retrieval_weights = np.concatenate(all_retrieval_weights, axis=0)
    
    print("\n" + "="*80)
    print("原型检索统计")
    print("="*80)
    print(f"总样本数: {len(all_retrieved_indices)}")
    print(f"Top-K: {args.top_k}")
    
 
    unique_indices, counts = np.unique(all_retrieved_indices.flatten(), return_counts=True)
    top_10_idx = np.argsort(counts)[-10:][::-1]
    print("\n最常被检索的10个原型:")
    for i, idx in enumerate(top_10_idx):
        proto_id = unique_indices[idx]
        count = counts[idx]
        print(f"  {i+1}. 原型 #{proto_id}: 被检索 {count} 次 ({count/len(all_retrieved_indices)*100:.2f}%)")
    
    
    mean_weights = all_retrieval_weights.mean(axis=0)
    print(f"\n平均检索权重 (Top-{args.top_k}):")
    for i, w in enumerate(mean_weights):
        print(f"  Top-{i+1}: {w:.4f}")
    print("="*80)

   
    df = pd.DataFrame(all_slice_results)

  
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(args.work_dir, f"results_cpal_sam.csv")

    def summarize(sub_df, label, target_name):
        s = {}
        for m in metrics_list:
            vals = sub_df[m].dropna()
            s[m] = f"{vals.mean():.4f}±{vals.std():.4f}" if len(vals) > 0 else "N/A"
        print(f"\n=== CM-CPAL-SAM [{target_name}] {label} (n={len(sub_df)}) ===")
        for m in metrics_list:
            print(f"  {m}: {s[m]}")
        return s

    summary_rows = []
    for cat in ["core", "penumbra"]:
        df_cat = df[df["category"] == cat]
        if len(df_cat) == 0:
            continue
        if cat == "core":
            df_eval = df_cat[df_cat["gt_nonempty"]]
            eval_label = "GT-nonempty slices"
            subset_tag = "gt_nonempty"
        else:
            df_eval = df_cat
            eval_label = "All slices"
            subset_tag = "all"
        df_nonempty = df_cat[df_cat["nonempty"]]

        s_eval = summarize(df_eval, eval_label, cat)
        s_nonempty = summarize(df_nonempty, "Non-empty slices", cat)
        summary_rows.append({"Model": "CM-CPAL-SAM", "Target": cat, "Subset": subset_tag, **s_eval})
        summary_rows.append({"Model": "CM-CPAL-SAM", "Target": cat, "Subset": "nonempty", **s_nonempty})

    # Save per-slice CSV
    per_slice_csv = output_csv.replace(".csv", "_per_slice.csv")
    df.to_csv(per_slice_csv, index=False)
    print(f"\nPer-slice CSV: {per_slice_csv}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary CSV: {output_csv}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
