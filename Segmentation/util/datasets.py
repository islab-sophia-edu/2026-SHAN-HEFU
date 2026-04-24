# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class FlatImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform=None, nb_classes=1000):
        self.transform = transform
        self.nb_classes = nb_classes
        self.data_folder = data_folder

        self.image_paths = []
        if os.path.exists(self.data_folder):
            for f in os.listdir(self.data_folder):
                file_path = os.path.join(self.data_folder, f)
                if os.path.isfile(file_path) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(file_path)
        else:
            raise FileNotFoundError(f"Image directory not found: {self.data_folder}")

        self.labels = {}
        is_val_dataset = 'val' in os.path.basename(self.data_folder.rstrip(os.sep)).lower()

        if is_val_dataset:
            self._load_val_labels(self.data_folder)
        else:
            self._load_train_labels(self.data_folder) # Fallback for robustness

        if not self.labels and is_val_dataset:
            raise ValueError(f"Validation labels could not be loaded. Fine-tuning requires image labels. Check {os.path.dirname(self.data_folder)}/val_labels.json or its content.")
        elif not self.labels:
            pass # Expected if it's a non-val set primarily handled by ImageFolder

        initial_image_count = len(self.image_paths)
        self.image_paths = [p for p in self.image_paths if p in self.labels]

        print(f"Loaded {len(self.image_paths)} images with labels from {self.data_folder}. Initial images: {initial_image_count}")
        if len(self.image_paths) == 0:
            print(f"WARNING: No images found with labels in {self.data_folder}. Check paths and label files.")


    def _load_train_labels(self, current_data_folder):
        label_map_path = os.path.join(os.path.dirname(current_data_folder), 'train_labels.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                raw_list = json.load(f) # Expects list of lists, e.g., [["path", label], ...]
            
            self.labels = {}
            imagenet_root_path = os.path.dirname(current_data_folder)
            for relative_path, label_id in raw_list:
                image_full_path = os.path.join(imagenet_root_path, relative_path)
                self.labels[image_full_path] = label_id
            print(f"Successfully loaded {len(self.labels)} train labels from {label_map_path}.")
        else:
            print(f"WARNING: Train labels JSON not found at {label_map_path}. This is expected if the train set is handled by ImageFolder.")


    def _load_val_labels(self, current_data_folder):
        label_map_path = os.path.join(os.path.dirname(current_data_folder), 'val_labels.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f) # Now expects a dictionary, e.g., {"path": label, ...}
            
            self.labels = {}
            imagenet_root_path = os.path.dirname(current_data_folder)
            for relative_path, label_id in raw_dict.items():
                image_full_path = os.path.join(imagenet_root_path, relative_path)
                self.labels[image_full_path] = label_id
            print(f"Successfully loaded {len(self.labels)} validation labels from {label_map_path}.")
        else:
            print(f"WARNING: Validation labels JSON not found at {label_map_path}. Using dummy labels.")
            for i, img_path in enumerate(self.image_paths):
                self.labels[img_path] = i % self.nb_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.labels.get(img_path)
        if label is None:
            raise ValueError(f"Label not found for image: {img_path}. Check label loading logic.")

        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    # specific_data_path will be /media/data_hdd/shanhefu/data/train or /media/data_hdd/shanhefu/data/val
    specific_data_path = os.path.join(args.data_path, 'train' if is_train else 'val')

    if is_train:
        print(f"Building train dataset: Using torchvision.datasets.ImageFolder. Root: {specific_data_path}")
        dataset = datasets.ImageFolder(specific_data_path, transform=transform)
    else: # is_train == False, for validation set
        print(f"Building validation dataset: Using FlatImageNetDataset. Root: {specific_data_path}")
        dataset = FlatImageNetDataset(specific_data_path, transform=transform, nb_classes=args.nb_classes)

    print(f"Using dataset: {type(dataset).__name__} for {'train' if is_train else 'validation'} set.")

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)