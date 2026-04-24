import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from dataset_stanford_normal_6ch import build_normal_dataset

# 假设你的模型定义在这里，请根据实际情况修改 import
try:
    from models_normal_6ch import vit_huge_patch14 # 或者你的模型文件名
except:
    # 如果找不到，给我一个 dummy model 测试 pipeline
    print("Warning: Could not import model. Using Dummy Model.")
    class DummyModel(nn.Module):
        def __init__(self, img_size=64, patch_size=64, in_chans=6, num_classes=3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1)
            )
        def forward(self, x, angle):
            return self.conv(x)
    vit_huge_patch14 = DummyModel

# ------------------------------------------------------------------
# 1. 正确的 Loss 函数 (关键！)
# ------------------------------------------------------------------
class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, mask=None):
        # pred: [B, 3, H, W] (Logits, potentially unnormalized)
        # target: [B, 3, H, W] (Normalized vectors)
        
        # 1. 强制归一化预测值 (Normals must be unit vectors)
        pred_norm = F.normalize(pred, dim=1, p=2)
        
        # 2. Cosine Similarity: sum(a * b)
        # Target 已经是归一化的，Pred 也归一化了
        # cos_sim 范围 [-1, 1]。1 是完全重合，-1 是相反。
        cos_sim = torch.sum(pred_norm * target, dim=1) 
        
        # 3. Loss = 1 - cos_sim (范围 0 到 2)
        # 0 means perfect alignment.
        loss_pixel = 1.0 - cos_sim
        
        if mask is not None:
            # mask: [B, 1, H, W]
            mask = mask.float()
            loss = (loss_pixel * mask.squeeze(1)).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_pixel.mean()
            
        return loss

# ------------------------------------------------------------------
# 2. 主训练循环 (只跑一个 Batch)
# ------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Args
    args = argparse.Namespace()
    args.data_path = '/home/shanhefu/Stanford2D3D' 
    args.val_data_path = '/home/shanhefu/Stanford2D3D'
    args.grid_height = 16
    args.pano_h = 1024
    args.pano_w = 2048
    
    # 1. Data (Load ONE batch)
    print("Loading Dataset...")
    dataset = build_normal_dataset(is_train=True, args=args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get ONE batch and keep it
    data_iter = iter(loader)
    patches_img, angles, patches_norm, patches_mask = next(data_iter)
    
    patches_img = patches_img.to(device)   # [B, N, 6, H, W]
    patches_norm = patches_norm.to(device) # [B, N, 3, H, W]
    patches_mask = patches_mask.to(device) # [B, N, 1, H, W]
    
    # Flatten Batch and Patches for simple training
    # [B*N, C, H, W]
    B, N, C, H, W = patches_img.shape
    inputs = patches_img.view(B*N, C, H, W)
    targets = patches_norm.view(B*N, 3, H, W)
    masks = patches_mask.view(B*N, 1, H, W)
    
    print(f"Input Shape: {inputs.shape}")
    print(f"Target Shape: {targets.shape}")
    
    # 2. Model
    print("Initializing Model...")
    # 确保你的模型接受 in_chans=6
    try:
        model = vit_huge_patch14(img_size=64, patch_size=64, in_chans=6, output_size=(512, 1024))
    except:
        print("Using standard call...")
        model = vit_huge_patch14(img_size=64, patch_size=64) 
        # 如果模型第一层是3通道，这里会报错。你需要修改模型定义的 patch_embed in_chans=6
        
    model.to(device)
    model.train()
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = NormalLoss()
    
    print("\nStarting Overfit Test (Goal: Loss -> 0.00)")
    print("-" * 50)
    
    for epoch in range(201):
        optimizer.zero_grad()
        
        # Forward
        # 注意：你的模型 forward 接收什么参数？
        # 如果是 ViT，通常是 (x, angles) 或者只 (x)
        # 这里为了测试，我们假设只传 x (input patch)
        # 如果你的模型必须传 angles，请取消注释下一行
        # output = model(inputs, angles.repeat_interleave(N, dim=0)) 
        
        # 尝试只传 inputs (RGB+XYZ)
        output = model(inputs) 
        
        # 如果模型输出是 list/tuple，取第一个
        if isinstance(output, (list, tuple)):
            output = output[0]
            
        # 确保 Output 也是 [B*N, 3, H, W]
        if output.shape != targets.shape:
            output = output.view(B*N, 3, H, W)
            
        # Loss
        loss = criterion(output, targets, masks)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Iter {epoch}: Loss = {loss.item():.6f}")
            
            # Check Prediction Stats
            with torch.no_grad():
                pred_norm = F.normalize(output, dim=1)
                cos_sim = (pred_norm * targets).sum(1).mean().item()
                print(f"    -> Mean Cosine Similarity: {cos_sim:.4f} (Target: 1.0)")

    print("-" * 50)
    if loss.item() < 0.05:
        print("[SUCCESS] The pipeline works! The model CAN learn.")
        print("Dataset code is correct. The problem was likely Loss Function or Hyperparams.")
    else:
        print("[FAIL] The model cannot even memorize 2 images.")
        print("CHECK: 1. Model Input Layer (is it actually using 6 channels?)")
        print("       2. Loss Function Logic")
        print("       3. Gradient (is it None?)")

if __name__ == '__main__':
    main()