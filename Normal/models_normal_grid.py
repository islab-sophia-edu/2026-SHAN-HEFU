from normal_model_core import PanoNormal, Fixed2DSinusoidalPositionalEncoding


def _make(embed_dim, depth, num_heads, **kwargs):
    kwargs["use_gctt"] = True
    kwargs["use_angular_bias"] = True
    kwargs["use_gauge_bias"] = True
    kwargs["position_mode"] = "spherical"
    kwargs.setdefault("patch_size", 32)
    return PanoNormal(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)


def vit_base_patch16(**kwargs):
    return _make(768, 12, 12, **kwargs)


def vit_large_patch16(**kwargs):
    return _make(1024, 24, 16, **kwargs)


def vit_huge_patch14(**kwargs):
    return _make(1280, 32, 16, **kwargs)
