import importlib
from pathlib import Path
import normal_train_core as core

VARIANT = "pb"


def main(args):
    args.variant = VARIANT
    core.models_normal = importlib.import_module("models_normal_" + VARIANT)
    core.engine_normal = importlib.import_module("engine_normal_" + VARIANT)
    core.build_normal_dataset = importlib.import_module("dataset_stanford_normal_" + VARIANT).build_normal_dataset
    return core.main(args)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = core.get_args_parser().parse_args()
    args.variant = VARIANT
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
