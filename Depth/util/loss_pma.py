import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(root_dir, classes, device):
    """
    Computes SQRT-dampened class weights.
    Ignores dummy placeholders.
    Returns: Tensor [C]
    """
    print(f"Calculating weights from: {root_dir}")
    
    class_counts = []
    dummy_filename = "dummy_placeholder.jpg"
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    # 1. Count Valid Images
    for class_name in classes:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.exists(class_path):
            count = 0
        else:
            files = [f for f in os.listdir(class_path) if f.lower().endswith(valid_ext)]
            if dummy_filename in files:
                count = len(files) - 1 # Exclude dummy
            else:
                count = len(files)
        class_counts.append(max(0, count))

    # 2. Compute Soft Weights (SQRT Inverse)
    weights = []
    for count in class_counts:
        if count > 0:
            w = 1.0 / np.sqrt(count) # Dampening
        else:
            w = 0.0 # Ignore empty/dummy-only classes
        weights.append(w)
    
    weights = np.array(weights)

    # 3. Normalize & Clip
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        weights[valid_mask] /= weights[valid_mask].mean()
        weights = np.clip(weights, 0.0, 10.0) # Prevent explosion

    weights_tensor = torch.tensor(weights, device=device, dtype=torch.float)
    
    # Print stats
    print(f"Class Counts (Head): {class_counts[:5]}")
    print(f"Weights (Head): {weights_tensor.cpu().numpy().round(4)[:5]}")
    
    return weights_tensor

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss adapted for Class Imbalance + Mixup (Soft Targets).
    Formula: - sum( alpha * (1 - p)^gamma * target * log(p) )
    """
    def __init__(self, class_weights=None, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.gamma = gamma # Focusing parameter (gamma=2 is standard)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x: [B, C] (Logits)
        # target: [B, C] (Soft probabilities from Mixup)
        
        # Calculate probabilities
        probs = F.softmax(x, dim=-1) # [B, C]
        logprobs = F.log_softmax(x, dim=-1) # [B, C]
        
        # Focal Term: (1 - p)^gamma
        # If p is high (easy sample), (1-p) is small -> weight reduced.
        # If p is low (hard sample/rare class), (1-p) is large -> weight maintained.
        focal_term = (1 - probs).pow(self.gamma)
        
        # Basic Cross Entropy part: - target * log(p)
        loss = -target * logprobs
        
        # Combine: Focal * CE
        loss = focal_term * loss
        
        # Apply Class Balance Weights (Alpha)
        if self.class_weights is not None:
            # Broadcast: [C] -> [1, C]
            loss = loss * self.class_weights.unsqueeze(0)
            
        # Sum over classes, mean over batch
        return loss.sum(dim=-1).mean()