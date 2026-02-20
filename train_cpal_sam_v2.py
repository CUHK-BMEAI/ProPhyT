"""
将S预训练的NCCT encoder和CTP原型库集成到SAM-Med2D训练中

是ctp based训练代码，这里只看dense_prompt模式即可
支持三种融合模式的测试：cross_attention / dense_prompt / sparse_prompt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import datetime
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import json


from segment_anything import sam_model_registry


from DataLoader import TrainingDataset, TestingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger
from metrics import SegMetrics


from cpal_sam_modules import build_cpal_sam_model


def parse_args():
    parser = argparse.ArgumentParser(description='CM-CPAL SAM Training v2')

    parser.add_argument('--work_dir', type=str, default='workdir',
                       help='工作目录')
    parser.add_argument('--run_name', type=str, default='cpal_sam_v2',
                       help='运行名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    parser.add_argument('--data_path', type=str, 
                       default='data_penumbra_noblank_withvalid',
                       help='数据路径')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='batch size')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像尺寸')
    parser.add_argument('--mask_num', type=int, default=2,
                       help='每个图像的mask数量')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')


    parser.add_argument('--epochs', type=int, default=15,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                       help='学习率调度器')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--val_interval', type=int, default=50,
                       help='每多少个batch进行一次验证')

 
    parser.add_argument('--model_type', type=str, default='vit_b',
                       help='SAM模型类型')
    parser.add_argument('--sam_checkpoint', type=str, 
                       default='sam-med2d_b1106.pth',
                       help='SAM预训练checkpoint')
    parser.add_argument('--multimask', action='store_true', default=True,
                       help='输出多个mask')
    parser.add_argument('--encoder_adapter', action='store_true', default=True,
                       help='使用adapter')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['iou', 'dice'],
                       help='评估指标')

    parser.add_argument('--stage1b_ckpt', type=str,
                       default='scripts/logs/cpal_stage1b_v2/best_model.pth',
                       help='ncct encoder checkpoint')
    parser.add_argument('--prototype_bank', type=str,
                       default='scripts/logs/cpal_stage1a_post/ctp_prototypes_post.npy',
                       help='CTP prototypes bank')
    parser.add_argument('--fusion_mode', type=str, default='cross_attention',
                       choices=['cross_attention', 'dense_prompt', 'sparse_prompt'],
                       help='fusion mode for ctp proto and img feature')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='dim for portotype')
    parser.add_argument('--top_k', type=int, default=5,
                       help='always 5')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='temparture for retrieval')
    parser.add_argument('--freeze_ncct', action='store_true',
                       help='frezzed ncct encoder')
    parser.add_argument('--freeze_sam_encoder', action='store_true',
                       help='freeze sam encoder')
    parser.add_argument('--random_prototype', action='store_true',
                       help='random')
    parser.add_argument('--hard_weight', action='store_true',
                       help='hard weight')

    args = parser.parse_args()
    
  
    if args.resume is not None:
        args.sam_checkpoint = None
    
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key in ['image', 'label']:
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


class CPALSAMTrainer:
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
       
        set_seed(args.seed)
        
   
        self.model_dir = os.path.join(args.work_dir, 'models', args.run_name)
        self.log_dir = os.path.join(args.work_dir, 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
       
        log_file = os.path.join(
            self.log_dir,
            f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"
        )
        self.logger = get_logger(log_file)
        
     
        self._save_config()
        
    
        self.model = self._build_model()
  
        self.optimizer = self._build_optimizer()
        
    
        self.scheduler = self._build_scheduler()
       
        self.criterion = FocalDiceloss_IoULoss()
        
     
        if args.use_amp:
            try:
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level="O1"
                )
                self.logger.info("*******Mixed precision with Apex enabled")
                self.use_amp = True
            except ImportError:
                self.logger.info("*******Apex not available, using FP32")
                self.use_amp = False
        else:
            self.use_amp = False
            self.logger.info("*******Using FP32 training")
        
     
        self.train_loader, self.valid_loader = self._build_dataloaders()

      
        self.best_valid_loss = float('inf')
        self.start_epoch = 0
        self.global_step = 0  

        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _save_config(self):
        config_path = os.path.join(self.model_dir, 'config.json')
        config = vars(self.args)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Config saved to {config_path}")
    
    def _build_model(self):
      
        self.logger.info("="*60)
        self.logger.info("Building CM-CPAL SAM Model v2")
        self.logger.info("="*60)
        

        sam_model = sam_model_registry[self.args.model_type](self.args)
        sam_model = sam_model.to(self.device)
        self.logger.info(f"✓ Loaded SAM model: {self.args.model_type}")
        
      
        cpal_sam_model = build_cpal_sam_model(
            sam_model=sam_model,
            stage1b_ckpt_path=self.args.stage1b_ckpt,
            prototype_bank_path=self.args.prototype_bank,
            fusion_mode=self.args.fusion_mode,
            latent_dim=self.args.latent_dim,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
            freeze_ncct=self.args.freeze_ncct,
            random_selection=self.args.random_prototype,
            hard_weight=self.args.hard_weight,
        )
        
        self.logger.info(f"✓ Built CM-CPAL SAM model v2")
        self.logger.info(f"  - Stage 1b V2: {self.args.stage1b_ckpt}")
        self.logger.info(f"  - Prototype Bank: {self.args.prototype_bank}")
        self.logger.info(f"  - Fusion mode: {self.args.fusion_mode}")
        self.logger.info(f"  - Top-K: {self.args.top_k}")
        self.logger.info(f"  - Temperature: {self.args.temperature}")
        self.logger.info(f"  - Freeze NCCT: {self.args.freeze_ncct}")
        
    
        if self.args.freeze_sam_encoder:
            for param in cpal_sam_model.sam_model.image_encoder.parameters():
                param.requires_grad = False
            self.logger.info("✓ Froze SAM image encoder")
        else:
            for n, param in cpal_sam_model.sam_model.image_encoder.named_parameters():
                if "Adapter" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.logger.info("✓ SAM image encoder: only Adapters trainable")

      
        for param in cpal_sam_model.sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        self.logger.info("✓ Froze SAM prompt_encoder (mask_decoder remains trainable)")

  
        self._print_param_stats(cpal_sam_model)
        
        return cpal_sam_model
    
    def _print_param_stats(self, model):
      
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info("-"*60)
        self.logger.info("Model Parameters:")
        self.logger.info(f"  Total: {total_params/1e6:.2f}M ({total_params:,})")
        self.logger.info(f"  Trainable: {trainable_params/1e6:.2f}M ({trainable_params:,})")
        self.logger.info(f"  Frozen: {(total_params-trainable_params)/1e6:.2f}M")
        
       
        ncct_params = sum(p.numel() for p in model.ncct_feature_extractor.parameters() if p.requires_grad)
        retriever_params = sum(p.numel() for p in model.prototype_retriever.parameters() if p.requires_grad)
        sam_params = sum(p.numel() for p in model.sam_model.parameters() if p.requires_grad)
        
        self.logger.info(f"  - NCCT Encoder: {ncct_params/1e6:.2f}M ({'trainable' if ncct_params > 0 else 'frozen'})")
        self.logger.info(f"  - Prototype Retriever: {retriever_params/1e6:.2f}M")
        self.logger.info(f"  - SAM Model: {sam_params/1e6:.2f}M")
        
        if self.args.fusion_mode == 'cross_attention' and model.cross_attention_fusion:
            fusion_params = sum(p.numel() for p in model.cross_attention_fusion.parameters() if p.requires_grad)
            self.logger.info(f"  - Cross Attention Fusion: {fusion_params/1e6:.2f}M")
        elif model.prompt_replacer:
            replacer_params = sum(p.numel() for p in model.prompt_replacer.parameters() if p.requires_grad)
            self.logger.info(f"  - Prompt Replacer: {replacer_params/1e6:.2f}M")
        
        self.logger.info("-"*60)
    
    def _build_optimizer(self):
        """构建优化器"""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr
        )
        self.logger.info(f"✓ Optimizer: Adam (lr={self.args.lr})")
        return optimizer
    
    def _build_scheduler(self):
        if self.args.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[5, 10], gamma=0.5
            )
            self.logger.info("✓ Scheduler: MultiStepLR")
            return scheduler
        return None
    
    def _build_dataloaders(self):
        # 训练集
        train_dataset = TrainingDataset(
            self.args.data_path,
            image_size=self.args.image_size,
            mode='train',
            point_num=1,
            mask_num=1,  # 简化：每个图像只生成1个mask
            requires_name=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        self.logger.info(f"✓ Train dataset: {len(train_dataset)} samples")
        
        
        try:
            valid_dataset = TestingDataset(
                self.args.data_path,
                image_size=self.args.image_size,
                mode='valid',
                requires_name=False,
                point_num=1,
                return_ori_mask=False,
                prompt_path=None
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
            self.logger.info(f"✓ Valid dataset: {len(valid_dataset)} samples")
        except Exception as e:
            self.logger.warning(f"Failed to load valid dataset: {e}")
            valid_loader = None
        
        return train_loader, valid_loader
    
    def _load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
        
        self.logger.info(f"✓ Resumed from epoch {self.start_epoch}")
        self.logger.info(f"✓ Best valid loss: {self.best_valid_loss:.4f}")
    
    def train_one_epoch(self, epoch):
        """训练一个epoch,无迭代训练"""
        self.model.train()
        train_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, batched_input in enumerate(pbar):
            # 数据预处理
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, self.device)

            images = batched_input["image"]
            labels = batched_input["label"]

            # 随机选择prompt类型
            if random.random() > 0.5:
                batched_input["point_coords"] = None
                prompt_type = "boxes"
            else:
                batched_input["boxes"] = None
                prompt_type = "point"

          
            masks, iou_predictions, retrieval_info = self.model(
                images=images,
                point_coords=batched_input.get("point_coords"),
                point_labels=batched_input.get("point_labels"),
                boxes=batched_input.get("boxes"),
                mask_inputs=batched_input.get("mask_inputs")
            )

           
            loss = self.criterion(masks, labels, iou_predictions)

          
            if self.use_amp:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

   
            train_losses.append(loss.item())

         
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        
            self.global_step += 1
            if (batch_idx + 1) % self.args.val_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

            
                valid_loss, valid_dice = self.validate(epoch)
                if valid_loss is not None:
                    self.logger.info(
                        f"Validation at Step {self.global_step}: "
                        f"Loss={valid_loss:.4f}, Dice={valid_dice:.4f}"
                    )

                 
                    is_best = False
                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        is_best = True

                    self.save_checkpoint(epoch, batch_idx=batch_idx, is_best=is_best)


        avg_loss = np.mean(train_losses)

        return avg_loss

    def validate(self, epoch):
        if self.valid_loader is None:
            return None, None
        
        self.model.eval()
        valid_losses = []
        valid_dice_scores = []
        
        with torch.no_grad():
            for batched_input in tqdm(self.valid_loader, desc="Validating"):
                batched_input = to_device(batched_input, self.device)
                
                images = batched_input["image"]
                labels = batched_input["label"]
                
            
                masks, iou_predictions, _ = self.model(
                    images=images,
                    point_coords=batched_input.get("point_coords"),
                    point_labels=batched_input.get("point_labels"),
                    boxes=batched_input.get("boxes")
                )
                
        
                loss = self.criterion(masks, labels, iou_predictions)
                valid_losses.append(loss.item())
                
                batch_dice = SegMetrics(masks, labels, ['dice'])
                valid_dice_scores.extend(batch_dice)
        
        avg_loss = np.mean(valid_losses)
        avg_dice = np.mean(valid_dice_scores) if valid_dice_scores else 0.0
        
        self.model.train()
        return avg_loss, avg_dice
    
    def save_checkpoint(self, epoch, batch_idx=None, is_best=False):
        state = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'args': vars(self.args)
        }

      
        latest_path = os.path.join(self.model_dir, 'latest.pth')
        torch.save(state, latest_path)

       
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(state, best_path)
            step_info = f"Step {self.global_step}" if batch_idx is not None else f"Epoch {epoch+1}"
            self.logger.info(f"✓ Saved best model at {step_info} (valid_loss={self.best_valid_loss:.4f})")
    
    def train(self):

        self.logger.info("="*60)
        self.logger.info("Starting Training (Simplified, No Iteration)")
        self.logger.info(f"Validation every {self.args.val_interval} batches")
        self.logger.info("="*60)

        for epoch in range(self.start_epoch, self.args.epochs):
            start_time = time.time()

            
            train_loss = self.train_one_epoch(epoch)

            
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.args.lr

            
            epoch_time = time.time() - start_time

            log_msg = (f"Epoch {epoch+1}/{self.args.epochs} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"LR: {current_lr:.6f} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Best Valid Loss: {self.best_valid_loss:.4f}")

            self.logger.info(log_msg)

        self.logger.info("="*60)
        self.logger.info("Training Completed!")
        self.logger.info(f"Best Valid Loss: {self.best_valid_loss:.4f}")
        self.logger.info("="*60)


def main():
    args = parse_args()
    trainer = CPALSAMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
