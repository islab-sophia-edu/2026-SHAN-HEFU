"""Layer-wise LR decay for PanoMAE downstream models.

Unlike the original MAE helper, PanoMAE has `view_embed` and spherical
position/attention modules. These pretrained modules must not be placed in
the task-head learning-rate group.
"""


def get_layer_id(name: str, num_layers: int) -> int:
    stem_prefixes = (
        "view_embed.",
        "patch_embed.",
        "angle_pos_embed.",
        "enc_ang_bias.",
        "ray_token_embed.",
    )
    if name in {"cls_token", "pos_embed", "ray_embed_scale"} or name.startswith(stem_prefixes):
        return 0
    if name.startswith("blocks."):
        index = int(name.split(".")[1])
        if not 0 <= index < num_layers - 1:
            raise ValueError(f"Invalid encoder block name: {name}")
        return index + 1
    return num_layers


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=(), layer_decay=0.75):
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError(f"layer_decay must be in (0,1], got {layer_decay}")
    num_layers = len(model.blocks) + 1
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    no_decay_names = set(no_weight_decay_list)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name, num_layers)
        no_decay = param.ndim == 1 or name in no_decay_names
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {
                "lr_scale": scales[layer_id],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "params": [],
            }
        groups[key]["params"].append(param)
    return list(groups.values())


def audit_stem_lr(model, optimizer, layer_decay, encoder_lr_mult=1.0):
    """Fail fast if the pretrained stem is silently trained as a task head."""
    expected = layer_decay ** (len(model.blocks) + 1) * encoder_lr_mult
    names = {id(p): name for name, p in model.named_parameters()}
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = names[id(param)]
            if name.startswith(("view_embed.", "angle_pos_embed.", "enc_ang_bias.")):
                seen.add(name.split(".")[0])
                actual = float(group.get("lr_scale", 1.0))
                if abs(actual - expected) > max(1e-12, abs(expected) * 1e-6):
                    raise RuntimeError(
                        f"Pretrained stem has incorrect LR scale: {name}: "
                        f"actual={actual:.9g}, expected={expected:.9g}"
                    )
    expected_stems = {"view_embed"}
    if any(name.startswith("angle_pos_embed.") and p.requires_grad
           for name, p in model.named_parameters()):
        expected_stems.add("angle_pos_embed")
    if not expected_stems.issubset(seen):
        raise RuntimeError(f"Missing pretrained stem groups in optimizer: {seen}")
    print(f"[PanoMAE LRD] pretrained stem LR scale={expected:.6g}; "
          f"task-head LR scale=1; encoder blocks={len(model.blocks)}")
