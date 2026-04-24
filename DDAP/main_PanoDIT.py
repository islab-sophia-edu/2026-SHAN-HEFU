import argparse
import os
import time
import datetime
import json
import torch
from diffusers import DDPMScheduler, AutoencoderKL
import torch.multiprocessing as mp

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
# ✅ 引入 MAE 官方的 Layer Decay
import util.lr_decay as lrd 
from util.datasets_PanoMAE import PanoramicDataset
from models_PanoDIT import PanoDiT, TextEmbedder
from engine_PanoDIT import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('Pano-DiT Fine-tuning', add_help=False)
    
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=32, type=int)
    
    # ✅ 加入 Layer Decay 支持
    parser.add_argument('--layer_decay', type=float, default=0.65, help='layer-wise lr decay from MAE')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.03)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    
    # 几何与尺寸参数
    parser.add_argument('--pano_h', default=1024, type=int)
    parser.add_argument('--pano_w', default=2048, type=int)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    
    # 数据集路径
    parser.add_argument('--data_path', default='/home/shanhefu/hybrid/train', type=str)
    # ✅ 加入验证集路径
    parser.add_argument('--val_data_path', default='/home/shanhefu/hybrid/test', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # 模型检查点
    parser.add_argument('--panomae_ckpt', default='', type=str, help='Pretrained PanoMAE checkpoint')
    return parser

def main(args):
    device = torch.device('cuda')
    
    # 1. 冻结的 VAE 和 CLIP
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=torch.bfloat16  # <--- 新增：半精度
    ).to(device)
    
    vae.enable_slicing()  # <--- 新增：Diffusers 原生防 OOM 机制，会在内部按 batch 自动切片
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    
    text_encoder = TextEmbedder().to(device)
    text_encoder.eval()
    
    # 2. Dataset 初始化
    dataset_train = PanoramicDataset(
        root_dir=args.data_path, grid_height=args.grid_height,
        pano_h=args.pano_h, pano_w=args.pano_w, img_size=args.patch_size, is_training=True
    )
    dataset_val = PanoramicDataset(
        root_dir=args.val_data_path, grid_height=args.grid_height,
        pano_h=args.pano_h, pano_w=args.pano_w, img_size=args.patch_size, is_training=False
    )
    
    spawn_ctx = mp.get_context('spawn')

    # 将 spawn_ctx 传给 DataLoader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        drop_last=True,
        multiprocessing_context=spawn_ctx
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        multiprocessing_context=spawn_ctx
    )

    # 提前获取纬度权重给 Loss 使用
    dummy_views, dummy_angles, dummy_weights = dataset_train[0]
    lat_weights = dummy_weights.to(device)

    # 3. 初始化 PanoDiT 模型
    latent_patch_size = args.patch_size // 8
    model = PanoDiT(
        latent_channels=4, latent_patch_size=latent_patch_size, 
        embed_dim=1280, depth=32, num_heads=16, grid_height=args.grid_height # Huge 配置
    ).to(device)
    
    if args.panomae_ckpt:
        model.load_from_panomae_checkpoint(args.panomae_ckpt)

    # ✅ 4. 使用 MAE 官方的逐层衰减 (Layer Decay) 构建优化器组
    param_groups = lrd.param_groups_lrd(
        model, args.weight_decay, 
        no_weight_decay_list=model.no_weight_decay() if hasattr(model, 'no_weight_decay') else {}, 
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # --- 训练主循环 ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, vae, noise_scheduler, text_encoder, dataloader_train, 
            optimizer, device, epoch, loss_scaler, lat_weights, args
        )
        
        # ✅ 每 eval_freq 或最后一轮，进行模型验证和生成出图
        if args.val_data_path and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            test_stats = evaluate(model, vae, text_encoder, dataloader_val, device, epoch, args)
            
            # 保存 checkpoint
            if args.output_dir:
                misc.save_model(args, model, model, optimizer, loss_scaler, epoch)
        
    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)