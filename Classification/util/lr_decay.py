# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    # 1. Blocks (Transformer Layers)
    if name.startswith("blocks"):
        return int(name.split('.')[1]) + 1
    
    # 2. Embeddings (Bottom Layer / Layer 0)
    # PanoMAE specific: view_embed, angle_pos_embed
    # Standard ViT: patch_embed, pos_embed
    elif name.startswith("patch_embed") or name.startswith("pos_embed") or \
         name.startswith("view_embed") or name.startswith("angle_pos_embed"):
        return 0
    
    # 3. CLS Token (Layer 0)
    elif name.startswith("cls_token"):
        return 0
    
    # 4. Head / Final Norm (Top Layer)
    elif name.startswith("norm") or name.startswith("head") or name.startswith("decoder"):
        return num_layers
    
    else:
        # Default to top layer (high LR) if unknown
        # print(f"WARNING: Assigning {name} to top layer (no decay).")
        return num_layers