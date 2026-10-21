"""Validate pretraining mask metadata without conflating two masking mechanisms."""


def audit_pretrain_masking(variant, saved_args):
    def get(key):
        return saved_args.get(key) if isinstance(saved_args, dict) else getattr(saved_args, key, None)

    adaptive = get("adaptive_masking")
    dynamic = get("dynamic_mask_ratio")
    ratio = get("mask_ratio")
    print(f"[Mask metadata] variant={variant}, adaptive_masking={adaptive}, "
          f"dynamic_mask_ratio={dynamic}, mask_ratio={ratio}")

    if variant == "grid":
        # main_pretrain_PanoMAE_grid.py defaults to non-adaptive spatial sampling.
        # engine_pretrain_PanoMAE_grid.py independently samples ratio U(0.6, 0.9).
        if adaptive is True:
            raise RuntimeError("Grid reference checkpoint expected uniform spatial masking "
                               "(adaptive_masking=False); found adaptive_masking=True")
        if dynamic is False:
            raise RuntimeError("Grid reference checkpoint expected --dynamic_mask_ratio "
                               "([0.6, 0.9]); checkpoint recorded False")
        if adaptive is None or dynamic is None:
            print("[Mask metadata WARNING] Grid masking flag missing; "
                  "confirm its pretraining command or log before reporting results")

    elif variant == "noadaptive":
        # Original noadaptive pretrain hard-codes mask_ratio=0.75 in the engine.
        # Its parser accepts --dynamic_mask_ratio but deliberately ignores it.
        if adaptive is True:
            raise RuntimeError("NoAdaptive checkpoint reports adaptive_masking=True")
        if dynamic is True:
            print("[Mask metadata] NoAdaptive pretrain ignored --dynamic_mask_ratio; "
                  "engine hard-codes 0.75. The flag in saved args does not change training.")
        if ratio is not None and abs(float(ratio) - 0.75) > 1e-6:
            print("[Mask metadata WARNING] NoAdaptive stored mask_ratio != 0.75, "
                  "but its original engine hard-codes current_mask_ratio=0.75")
        if adaptive is None:
            print("[Mask metadata WARNING] NoAdaptive adaptive_masking flag missing")

    elif variant == "gctt":
        if adaptive is False:
            raise RuntimeError("GCTT checkpoint reports adaptive_masking=False; "
                               "check its intended content/latitude-aware masking variant")
        if adaptive is None:
            print("[Mask metadata WARNING] GCTT adaptive_masking flag missing")

    # Other variants retain the original non-intrusive validation policy.
