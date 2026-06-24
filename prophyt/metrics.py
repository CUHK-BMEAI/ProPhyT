import torch
import numpy as np
import cv2

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])

    gt_sum = torch.sum(gt_, dim=[1,2,3])
    pr_sum = torch.sum(pr_, dim=[1,2,3])
    gt_has = gt_sum > 0
    pr_has = pr_sum > 0

    base_dice = (2. * intersection + eps) / (union + eps)
    # 规则：
    # pr>0 & gt>0 -> 正常dice
    # pr>0 & gt==0 -> 0
    # pr==0 & gt>0 -> 0
    # pr==0 & gt==0 -> 1
    both_zero = (~pr_has) & (~gt_has)
    both_pos = pr_has & gt_has
    dice_scores = torch.zeros_like(base_dice)
    dice_scores = torch.where(both_pos, base_dice, dice_scores)
    dice_scores = torch.where(both_zero, torch.ones_like(dice_scores), dice_scores)
    return dice_scores.cpu().numpy()


def dice_by_category(pr, gt, mask_paths=None, eps=1e-7, threshold=0.5):
    """
    分别计算core和penumbra mask的dice
    包含所有mask（包括空mask，空mask的dice设为0.0，与总体dice一致）
    
    参数:
        pr: 预测mask
        gt: 真实mask
        mask_paths: mask路径列表，用于判断是core还是penumbra
        eps: 平滑项
        threshold: 阈值
    
    返回:
        dict: {
            'core_dice': core mask的dice列表（包含所有core mask，空mask的dice为0.0）,
            'penumbra_dice': penumbra mask的dice列表（包含所有penumbra mask，空mask的dice为0.0）,
            'core_count': 所有core mask数量,
            'penumbra_count': 所有penumbra mask数量,
            'core_empty_count': 全空的core mask数量,
            'penumbra_empty_count': 全空的penumbra mask数量
        }
    """
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)

    intersection = torch.sum(gt_ * pr_, dim=[1,2,3])
    union = torch.sum(gt_, dim=[1,2,3]) + torch.sum(pr_, dim=[1,2,3])
    
    # 检查哪些mask是有前景的（非空的）
    gt_sum = torch.sum(gt_, dim=[1,2,3])
    has_foreground = gt_sum > 0
    pr_sum = torch.sum(pr_, dim=[1,2,3])
    pred_has_foreground = pr_sum > 0
    
    base_dice = (2. * intersection + eps) / (union + eps)
    # 按相同规则生成dice
    both_zero = (~pred_has_foreground) & (~has_foreground)
    both_pos = pred_has_foreground & has_foreground
    dice_scores = torch.zeros_like(base_dice)
    dice_scores = torch.where(both_pos, base_dice, dice_scores)
    dice_scores = torch.where(both_zero, torch.ones_like(dice_scores), dice_scores)
    
    # 如果没有提供mask_paths，则无法分类
    if mask_paths is None:
        return {
            'core_dice': np.array([]),
            'penumbra_dice': dice_scores.cpu().numpy(),
            'core_count': 0,
            'penumbra_count': int(len(dice_scores)),
            'core_empty_count': 0,  # 无分类信息时统计不到
            'penumbra_empty_count': int((~has_foreground).sum().item()),
            'core_nonempty_dice': np.array([]),
            'penumbra_nonempty_dice': dice_scores[has_foreground].cpu().numpy(),
            'core_empty_gt_pred_empty_count': 0,  # 无分类信息时统计不到
            'penumbra_empty_gt_pred_empty_count': int((~has_foreground & ~pred_has_foreground).sum().item()),
            'core_pred_nonempty_count': 0,  # 无分类信息时统计不到
            'penumbra_pred_nonempty_count': int(pred_has_foreground.sum().item())
        }
    
    # 判断每个mask是core还是penumbra
    # 根据路径或文件名中是否包含'core'或'penumbra'来判断
    is_core = []
    for path in mask_paths:
        path_lower = str(path).lower()
        # 检查路径中是否包含core关键字（但不包含penumbra）
        if 'core' in path_lower and 'penumbra' not in path_lower:
            is_core.append(True)
        else:
            # 默认认为是penumbra（如果路径中包含penumbra，或者都不包含）
            is_core.append(False)
    
    is_core = torch.tensor(is_core, device=dice_scores.device)
    is_penumbra = ~is_core
    
    # 包含所有mask（包括空mask）
    core_dice = dice_scores[is_core].cpu().numpy() if is_core.any() else np.array([])
    penumbra_dice = dice_scores[is_penumbra].cpu().numpy() if is_penumbra.any() else np.array([])
    
    # 统计空mask数量
    core_empty = is_core & (~has_foreground)
    penumbra_empty = is_penumbra & (~has_foreground)

    # 非空样本上的dice
    core_nonempty = is_core & has_foreground
    penumbra_nonempty = is_penumbra & has_foreground
    core_nonempty_dice = dice_scores[core_nonempty].cpu().numpy() if core_nonempty.any() else np.array([])
    penumbra_nonempty_dice = dice_scores[penumbra_nonempty].cpu().numpy() if penumbra_nonempty.any() else np.array([])

    # 空GT且预测为空（true negative）
    core_empty_gt_pred_empty = core_empty & (~pred_has_foreground)
    penumbra_empty_gt_pred_empty = penumbra_empty & (~pred_has_foreground)

    # 预测为非空的数量（用于检测“全黑”倾向）
    core_pred_nonempty = is_core & pred_has_foreground
    penumbra_pred_nonempty = is_penumbra & pred_has_foreground

    return {
        'core_dice': core_dice,
        'penumbra_dice': penumbra_dice,
        'core_count': int(is_core.sum().item()),
        'penumbra_count': int(is_penumbra.sum().item()),
        'core_empty_count': int(core_empty.sum().item()),
        'penumbra_empty_count': int(penumbra_empty.sum().item()),
        'core_nonempty_dice': core_nonempty_dice,
        'penumbra_nonempty_dice': penumbra_nonempty_dice,
        'core_empty_gt_pred_empty_count': int(core_empty_gt_pred_empty.sum().item()),
        'penumbra_empty_gt_pred_empty_count': int(penumbra_empty_gt_pred_empty.sum().item()),
        'core_pred_nonempty_count': int(core_pred_nonempty.sum().item()),
        'penumbra_pred_nonempty_count': int(penumbra_pred_nonempty.sum().item())
    }


def SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric