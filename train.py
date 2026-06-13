import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from prophyt.segment_anything import sam_model_registry, SamPredictor
from prophyt.data import TrainingDataset, TestingDataset, stack_dict_batched
from prophyt.utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from prophyt.metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=2, help="get mask number")
    parser.add_argument("--data_path", type=str, default=r"data_penumbra_noblank_withvalid", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="sam-med2d_b1106.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=str, default="False", help="use amp")

    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
  
    args.use_amp = args.use_amp.lower() in ['true', '1', 'yes']
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


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False,
                       B=None, mask_num=None):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    mask_inputs = batched_input.get("mask_inputs", None)
    if mask_inputs is not None:
        target_h, target_w = model.prompt_encoder.mask_input_size
        if mask_inputs.shape[-2] != target_h or mask_inputs.shape[-1] != target_w:
            batched_input["mask_inputs"] = F.interpolate(
                mask_inputs,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )


    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
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

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


def validate(args, model, valid_loader, criterion):

    model.eval()

    valid_losses = []
    valid_dice_scores = []

    with torch.no_grad():
        for batch_data in tqdm(valid_loader, desc="Validating"):

            batched_input = batch_data


            if batched_input["image"].dim() == 3:
                batched_input = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and k in ["image", "label"] and v.dim() == 3 else v)
                                for k, v in batched_input.items()}

            batched_input = to_device(batched_input, args.device)

            #get image embeddings
            if args.use_amp:
                labels = batched_input["label"].half()
                image_embeddings = model.image_encoder(batched_input["image"].half())
            else:
                labels = batched_input["label"]
                image_embeddings = model.image_encoder(batched_input["image"])

            # inference
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings,
                decoder_iter=True, B=image_embeddings.shape[0], mask_num=1
            )

      
            loss = criterion(masks, labels, iou_predictions)
            valid_losses.append(loss.item())

       
            batch_dice = SegMetrics(masks, labels, ['dice'])
            valid_dice_scores.extend(batch_dice)

    avg_loss = np.mean(valid_losses)
    avg_dice = np.mean(valid_dice_scores) if len(valid_dice_scores) > 0 else 0.0

    model.train()

    return avg_loss, avg_dice


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, valid_loader=None):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    valid_results = []
    # 如果使用 amp，在函数开始处导入
    amp = None
    if args.use_amp:
        try:
            from apex import amp
        except ImportError:
            args.use_amp = False
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())

            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings,
                decoder_iter=False, B=B, mask_num=args.mask_num
            )
            loss = criterion(masks, labels, iou_predictions)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            #这个生成的low_res_masks,用于生成point
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings,
                decoder_iter=False, B=B, mask_num=args.mask_num
            )
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

            # 每50个batch后运行验证
            if valid_loader is not None:
                valid_loss, valid_dice = validate(args, model, valid_loader, criterion)
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}')
                valid_results.append((batch+1, valid_loss, valid_dice))
                train_loader.set_postfix(
                    train_loss=loss.item(),
                    valid_loss=valid_loss,
                    valid_dice=valid_dice,
                    gpu_info={'gpu_name': args.device}
                )

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()

        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            if args.use_amp:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings,
                    decoder_iter=True, B=B, mask_num=args.mask_num
                )
                loss = criterion(masks, labels, iou_predictions)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings,
                    decoder_iter=True, B=B, mask_num=args.mask_num
                )
                loss = criterion(masks, labels, iou_predictions)
                loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

            if int(batch+1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')


        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics, valid_results


def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    criterion = FocalDiceloss_IoULoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"*******load {args.resume}")

    if args.use_amp:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            print("*******Mixed precision with Apex")
        except ImportError:
            print("*******Apex not available, falling back to no mixed precision")
            args.use_amp = False
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))

    # 加载验证集
    valid_loader = None
    try:
        valid_dataset = TestingDataset(args.data_path, image_size=args.image_size, mode='valid', requires_name=False, point_num=1, return_ori_mask=False, prompt_path=None)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print('*******Valid data:', len(valid_dataset))
    except Exception as e:
        print(f'*******Warning: Failed to load valid dataset: {e}')
        print('*******Continuing without validation set')

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_valid_dice = -1.0  # 使用dice作为最佳指标，初始化为-1
    l = len(train_loader)

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics, valid_results = train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion, valid_loader=valid_loader
        )

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr


        valid_info = ""
        if valid_results:
            last_valid_batch, last_valid_loss, last_valid_dice = valid_results[-1]
            valid_info = f", Valid Loss: {last_valid_loss:.4f}, Valid Dice: {last_valid_dice:.4f}"
            loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}{valid_info}")

            if last_valid_dice > best_valid_dice:
                best_valid_dice = last_valid_dice

                old_best_path = os.path.join(args.work_dir, "models", args.run_name, "best_dice_sam.pth")
                if os.path.exists(old_best_path):
                    try:
                        os.remove(old_best_path)
                    except:
                        pass

                save_path = os.path.join(args.work_dir, "models", args.run_name, "best_dice_sam.pth")
                state = {
                    'model': model.float().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_valid_dice': best_valid_dice,
                    'train_loss': average_loss,
                    'train_metrics': train_metrics
                }
                torch.save(state, save_path)
                print(f"*******Saved best checkpoint with Valid Dice: {best_valid_dice:.4f} at {save_path}")

                if args.use_amp:
                    model = model.half()
        else:
            loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
