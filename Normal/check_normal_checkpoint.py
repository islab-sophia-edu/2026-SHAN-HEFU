# debug_weight_matching.py
import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append('.')

from Normal.models_normal_cnn import vit_huge_patch14

def detailed_weight_analysis():
    """详细分析预训练权重与PanoNormal模型的匹配情况"""
    
    checkpoint_path = "/media/data_hdd1/shanhefu/outputs/panomae_pretrain/indoor/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-240.pth"
    
    print("="*80)
    print("权重匹配详细分析")
    print("="*80)
    
    # 1. 加载预训练权重
    print("\n[1] 加载预训练checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
        print(f"使用'model'键下的权重")
    else:
        pretrained_state_dict = checkpoint
        print(f"使用顶层权重")
    
    print(f"预训练权重键数量: {len(pretrained_state_dict)}")
    
    # 打印前20个键名
    print(f"\n预训练权重键名 (前20个):")
    for i, key in enumerate(list(pretrained_state_dict.keys())[:20]):
        shape = pretrained_state_dict[key].shape
        print(f"  {i:2d}. {key:50s} shape={shape}")
    
    # 2. 创建PanoNormal模型
    print("\n[2] 创建PanoNormal模型...")
    model = vit_huge_patch14(
        img_size=64,
        patch_size=64,
        grid_height=16,
        output_size=(1024, 2048),
        drop_path_rate=0.2,
    )
    
    model_state_dict = model.state_dict()
    print(f"模型参数键数量: {len(model_state_dict)}")
    
    print(f"\n模型键名 (前20个):")
    for i, key in enumerate(list(model_state_dict.keys())[:20]):
        shape = model_state_dict[key].shape
        print(f"  {i:2d}. {key:50s} shape={shape}")
    
    # 3. 键名匹配分析
    print("\n[3] 键名匹配分析...")
    
    # 尝试不同的键名转换策略
    transformation_strategies = [
        ("原始", lambda k: k),
        ("去除module.", lambda k: k.replace("module.", "")),
        ("去除encoder.", lambda k: k.replace("encoder.", "")),
        ("去除backbone.", lambda k: k.replace("backbone.", "")),
        ("去除所有前缀", lambda k: k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")),
        ("view_embed转view_embed.proj", lambda k: k.replace("view_embed.", "view_embed.proj.")),
        ("view_embed.proj转view_embed.proj.proj", lambda k: k.replace("view_embed.proj.", "view_embed.proj.proj.")),
    ]
    
    best_match = 0
    best_strategy = None
    best_mapping = {}
    
    for strategy_name, transform_fn in transformation_strategies:
        transformed_pretrained = {}
        for key, value in pretrained_state_dict.items():
            new_key = transform_fn(key)
            transformed_pretrained[new_key] = value
        
        # 计算匹配数
        matched_keys = []
        missing_keys = []
        for model_key in model_state_dict.keys():
            if model_key in transformed_pretrained:
                # 检查形状是否匹配
                pretrained_shape = transformed_pretrained[model_key].shape
                model_shape = model_state_dict[model_key].shape
                if pretrained_shape == model_shape:
                    matched_keys.append(model_key)
                else:
                    missing_keys.append(f"{model_key} (形状不匹配: {pretrained_shape} != {model_shape})")
            else:
                missing_keys.append(f"{model_key} (键不存在)")
        
        match_count = len(matched_keys)
        print(f"\n策略: {strategy_name:20s} 匹配数: {match_count}/{len(model_state_dict)} ({match_count/len(model_state_dict):.1%})")
        
        if match_count > best_match:
            best_match = match_count
            best_strategy = (strategy_name, transform_fn)
            best_mapping = transformed_pretrained
    
    # 4. 详细显示最佳策略的匹配情况
    print("\n" + "="*80)
    print(f"最佳策略: {best_strategy[0]}")
    print("="*80)
    
    transformed_pretrained = best_mapping
    
    # 分类显示匹配情况
    categories = {
        "view_embed相关": [],
        "blocks相关": [],
        "cls_token": [],
        "位置编码": [],
        "融合模块": [],
        "上采样": [],
        "CNN部分": [],
        "其他": []
    }
    
    for model_key in model_state_dict.keys():
        if model_key in transformed_pretrained:
            pretrained_shape = transformed_pretrained[model_key].shape
            model_shape = model_state_dict[model_key].shape
            status = "✅ 匹配" if pretrained_shape == model_shape else f"❌ 形状不匹配 ({pretrained_shape} != {model_shape})"
            
            # 分类
            if 'view_embed' in model_key:
                categories["view_embed相关"].append((model_key, status))
            elif 'blocks' in model_key:
                categories["blocks相关"].append((model_key, status))
            elif 'cls_token' in model_key:
                categories["cls_token"].append((model_key, status))
            elif 'pos' in model_key.lower() or 'angle' in model_key.lower():
                categories["位置编码"].append((model_key, status))
            elif 'fusion' in model_key:
                categories["融合模块"].append((model_key, status))
            elif 'up' in model_key or 'skip' in model_key:
                categories["上采样"].append((model_key, status))
            elif 'global_detail' in model_key or 'conv' in model_key.lower():
                categories["CNN部分"].append((model_key, status))
            else:
                categories["其他"].append((model_key, status))
        else:
            status = "❌ 缺失"
            
            # 分类
            if 'view_embed' in model_key:
                categories["view_embed相关"].append((model_key, status))
            elif 'blocks' in model_key:
                categories["blocks相关"].append((model_key, status))
            elif 'cls_token' in model_key:
                categories["cls_token"].append((model_key, status))
            elif 'pos' in model_key.lower() or 'angle' in model_key.lower():
                categories["位置编码"].append((model_key, status))
            elif 'fusion' in model_key:
                categories["融合模块"].append((model_key, status))
            elif 'up' in model_key or 'skip' in model_key:
                categories["上采样"].append((model_key, status))
            elif 'global_detail' in model_key or 'conv' in model_key.lower():
                categories["CNN部分"].append((model_key, status))
            else:
                categories["其他"].append((model_key, status))
    
    # 打印分类结果
    for category_name, items in categories.items():
        if items:
            print(f"\n{category_name} ({len(items)}个):")
            for key, status in items[:10]:  # 只显示前10个
                print(f"  {status:20s} {key}")
            if len(items) > 10:
                print(f"  ... 还有{len(items)-10}个")
    
    # 5. 关键的view_embed权重分析
    print("\n" + "="*80)
    print("关键的 view_embed 权重分析")
    print("="*80)
    
    # 查找预训练中的view_embed相关键
    pretrained_view_keys = [k for k in pretrained_state_dict.keys() if 'view' in k.lower() or 'patch' in k.lower() or 'embed' in k.lower()]
    
    print(f"\n预训练中的view/patch/embed相关键 ({len(pretrained_view_keys)}个):")
    for key in pretrained_view_keys:
        shape = pretrained_state_dict[key].shape
        print(f"  {key:60s} shape={shape}")
    
    # 模型中的view_embed键
    model_view_keys = [k for k in model_state_dict.keys() if 'view_embed' in k]
    
    print(f"\n模型中的view_embed键 ({len(model_view_keys)}个):")
    for key in model_view_keys:
        shape = model_state_dict[key].shape
        print(f"  {key:60s} shape={shape}")
    
    # 6. 建议的键名映射
    print("\n" + "="*80)
    print("建议的键名映射方案")
    print("="*80)
    
    # 基于分析，提供具体映射建议
    print("\n根据分析，建议在 main_finetune_normal.py 中使用以下映射:")
    
    mapping_examples = []
    
    # 检查可能的映射
    for model_key in model_view_keys[:5]:  # 检查前5个
        found = False
        for pretrained_key in pretrained_view_keys:
            # 尝试找到匹配
            if 'proj' in model_key and 'proj' in pretrained_key:
                # 检查形状
                if pretrained_state_dict[pretrained_key].shape == model_state_dict[model_key].shape:
                    mapping_examples.append((pretrained_key, model_key))
                    found = True
                    break
        
        if not found and 'weight' in model_key:
            # 尝试通用映射
            for pretrained_key in pretrained_view_keys:
                if 'weight' in pretrained_key and pretrained_state_dict[pretrained_key].shape == model_state_dict[model_key].shape:
                    mapping_examples.append((pretrained_key, model_key))
                    break
    
    if mapping_examples:
        print("\n示例映射:")
        for src, dst in mapping_examples:
            print(f"  '{src}' -> '{dst}'")
    else:
        print("\n⚠️ 无法自动找到映射，需要手动检查形状")
        
        # 显示形状对比
        print("\n形状对比:")
        for i, model_key in enumerate(model_view_keys[:3]):
            model_shape = model_state_dict[model_key].shape
            print(f"\n模型键: {model_key}")
            print(f"  形状: {model_shape}")
            print("  预训练中相似形状的键:")
            count = 0
            for pretrained_key in pretrained_view_keys:
                if pretrained_state_dict[pretrained_key].shape == model_shape:
                    print(f"    {pretrained_key}")
                    count += 1
                    if count >= 3:
                        break
            if count == 0:
                print("    (无匹配形状)")
    
    # 7. 创建修复的权重加载函数
    print("\n" + "="*80)
    print("修复的权重加载代码建议")
    print("="*80)
    
    print("""
def load_pretrained_weights_fixed(model, checkpoint_path):
    \"\"\"修复的权重加载函数\"\"\"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    
    new_state_dict = {}
    
    # 关键映射规则
    mapping_rules = [
        # Pano-MAE -> PanoNormal
        (r'^view_embed\.proj\.', 'view_embed.proj.proj.'),
        (r'^encoder\.view_embed\.', 'view_embed.proj.'),
        (r'^backbone\.view_embed\.', 'view_embed.proj.'),
        (r'^module\.view_embed\.', 'view_embed.proj.'),
        # blocks映射
        (r'^encoder\.blocks\.', 'blocks.'),
        (r'^backbone\.blocks\.', 'blocks.'),
        (r'^module\.blocks\.', 'blocks.'),
        # cls_token
        (r'^encoder\.cls_token', 'cls_token'),
        (r'^backbone\.cls_token', 'cls_token'),
        # norm
        (r'^encoder\.norm\.', 'norm.'),
    ]
    
    for k, v in state_dict.items():
        new_key = k
        
        # 移除常见前缀
        new_key = new_key.replace('module.', '').replace('encoder.', '').replace('backbone.', '')
        
        # 应用映射规则
        for pattern, replacement in mapping_rules:
            import re
            if re.match(pattern, new_key):
                new_key = re.sub(pattern, replacement, new_key)
                break
        
        # 跳过不需要的键
        if any(x in new_key for x in ['head', 'decoder', 'mask_token', 'pos_embed']):
            continue
            
        new_state_dict[new_key] = v
    
    # 尝试加载
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"加载结果: 缺失{len(msg.missing_keys)}个, 意外{len(msg.unexpected_keys)}个")
    
    # 检查关键组件
    critical_missing = [k for k in msg.missing_keys if 'view_embed' in k or 'blocks.0.' in k]
    if critical_missing:
        print(f"⚠️  关键组件缺失: {critical_missing[:5]}")
    
    return model
""")
    
    return best_match, len(model_state_dict)

if __name__ == "__main__":
    best_match, total = detailed_weight_analysis()
    print(f"\n总结: 最佳匹配率 {best_match}/{total} ({best_match/total:.1%})")
    
    if best_match / total < 0.5:
        print("\n⚠️  ⚠️  ⚠️  警告: 权重匹配率低于50%，模型可能无法有效利用预训练权重！")
        print("建议: 1. 检查预训练模型结构 2. 手动编写映射规则 3. 考虑从头训练")