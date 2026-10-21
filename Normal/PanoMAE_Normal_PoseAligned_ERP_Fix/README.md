# PanoMAE / GCTT — Stanford official Normal Pose alignment

## Verified geometry

40 train-area v2 audit panoramas: median signed official-normal-vs-XYZ error 0.63384° with `perm=0,2,1; sign=-1,-1,1`.

Calibration used 32 panoramas in areas 1–4. The fixed Euler `forward` basis was evaluated on **8 held-out training panoramas in area_6**, with median residual rotation 0.00168°, max 0.00548°. Pose inverse hypothesis: held-out median 99.78°.

The **test split area_5a is not part of this calibration**. The code uses its official pose without fitting to GT/model predictions. Good practice is to perform a read-only geometry validation on area_5a before reporting benchmark results.

Row-vector convention:

```python
R_erp_to_world = B @ (Rz(z) @ Ry(y) @ Rx(x)).T
n_erp = n_world @ R_erp_to_world.T
n_world = n_erp @ R_erp_to_world
```

`B = [[0,0,-1],[-1,0,0],[0,1,0]]`.

## Installation

Copy all files from `Normal/` into your **separate GCTT Normal experiment folder**, preserving your own original files/backups. Alternatively apply `patch -p1 < pose_alignment.patch` from the directory containing the original main/dataset/engine files from the candidate ZIP. `models_normal_gctt.py` and `normal_lrd.py` are unchanged.

A `pano/pose/*_pose.json` file must exist for **every** RGB/official normal pair. Do not open depth / global_xyz during training. The dataset now validates Pose paths while scanning samples and reuses Pose both for supervision and native GT visualization.

**Keep** `--normal_target_frame global`: this legacy code flag means *PanoMAE ERP-axis vector*, not Stanford world coordinates after this patch. `--normal_prediction_frame stanford` remains an identity transform in the engine. New `--official_normal_coordinate erp` documents the true supervision frame. For clarity in papers, call it *camera/ERP-aligned normals from official world GT*.

## Recommended isolated command

Use your previous launch arguments plus:

```bash
--normal_label_source normal \
--normal_target_frame global \
--normal_prediction_frame stanford \
--official_normal_coordinate erp \
--normal_axis_perm 0,2,1 \
--normal_axis_sign=-1,-1,1 \
--w_cos 1.0 \
--w_l1 0.0 \
--no_color_jitter \
--no_blur \
--no_horizontal_roll \
--gctt_gauge_jitter_deg 0 \
--gctt_local_gauge_jitter_deg 0 \
--lr 2e-5 \
--min_lr 1e-6 \
--save_normal_vis_every 1
```

`--lr 2e-5` is an initial controlled experiment, **not a demonstrated optimal LR**. Do not use `--resume` from pre-fix fine-tuned Normal checkpoints. Use only `--finetune` from a compatible GCTT **pretraining** checkpoint. Existing fine-tuned weights cannot be presumed correct in this new frame.

Run `python test_pose_frame.py --csv /path/to/audit_train_v2.csv` before training. Test requires NumPy, pandas, and Python's stdlib; it does not require CUDA or Stanford source images.

## Visualization, evaluation, fairness

Predicted and sampled GT patches are now both expressed in **ERP** axes before cosine loss, inverse-ODI and visual rendering. Engine's native GT display is also rotated from official **world** into ERP before comparison. No per-image best yaw or GT-dependent model-prediction transformation is used.

Official world-frame angular metrics are identical in theory to ERP-frame metrics **if both predictions and GT are rotated using the same official Pose**, because SO(3) preserves dot products. At deployment, ERP normals can be estimated from RGB only; world-frame normals additionally require camera Pose. Do not claim RGB-only global-world prediction.

If you want to show original official World GT, it must be in a **separately labeled panel**; do not compare it directly against ERP predictions.

This archive patches the standalone GCTT Normal candidate supplied previously, not all 6 ablations. Audit and test other branches before porting changes.
