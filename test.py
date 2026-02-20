from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics, dice_by_category
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
import re
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_erosion
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default=r"data_penumbra_noblank_withvalid", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=str, default="True", help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num")
    parser.add_argument("--multimask", type=str, default="True", help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=str, default="True", help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=str, default="False", help="save reslut")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--nssd_tau", type=float, default=2.0, help="NSD tolerance")

    args = parser.parse_args()
    args.boxes_prompt = args.boxes_prompt.lower() in ['true', '1', 'yes']
    args.multimask = args.multimask.lower() in ['true', '1', 'yes']
    args.encoder_adapter = args.encoder_adapter.lower() in ['true', '1', 'yes']
    args.save_pred = args.save_pred.lower() in ['true', '1', 'yes']
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=ddp_model.prompt_encoder.get_dense_pe(),
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
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def _get_surface_points(mask):
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
