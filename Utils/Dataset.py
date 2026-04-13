from torch.utils.data import Dataset
import cv2
import numpy
import albumentations as A
import timm
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from scipy.io import loadmat
import pkgutil
from timm.data import ImageNetInfo

def build_imagenet_label_mapping(devkit_dir: str):
    devkit_dir = Path(devkit_dir)

    # Parse meta.mat with attribute-style access
    meta = loadmat(
        str(devkit_dir / "data" / "meta.mat"),
        struct_as_record=False,
        squeeze_me=True,
    )
    synsets = meta["synsets"]

    # original devkit label id (1-based) -> wnid
    original_to_wnid = {}
    for s in synsets:
        ilsvrc_id = int(s.ILSVRC2012_ID)
        wnid = str(s.WNID)
        num_children = int(s.num_children)

        # Keep only ImageNet-1k leaf classes
        if num_children == 0:
            original_to_wnid[ilsvrc_id] = wnid

    if len(original_to_wnid) != 1000:
        raise ValueError(f"Expected 1000 leaf classes, got {len(original_to_wnid)}")

    # timm wnid order -> classifier index
    raw = pkgutil.get_data("timm", "data/_info/imagenet_synsets.txt")
    if raw is None:
        raise FileNotFoundError("timm imagenet_synsets.txt not found")

    timm_synsets = raw.decode("utf-8").strip().splitlines()
    wnid_to_timm_idx = {wnid: idx for idx, wnid in enumerate(timm_synsets)}

    # original devkit label id -> timm class index
    label_mapping = {}
    for original_label, wnid in original_to_wnid.items():
        if wnid not in wnid_to_timm_idx:
            raise KeyError(f"WNID {wnid} not found in timm synset list")
        label_mapping[original_label] = wnid_to_timm_idx[wnid]

    return label_mapping

def build_index_to_description(devkit_dir: str, label_mapping: dict):
    devkit_dir = Path(devkit_dir)
    meta = loadmat(str(devkit_dir / "data" / "meta.mat"), squeeze_me=True)
    synsets = meta["synsets"]

    # original devkit label (1-based) -> description
    original_label_to_description = {}
    for s in synsets:
        ilsvrc_id = int(s[0])
        words = str(s[2])
        num_children = int(s[4])
        if num_children == 0:
            original_label_to_description[ilsvrc_id] = words

    # timm index (mapped label) -> description
    index_to_description = {
        mapped_idx: original_label_to_description[original_label]
        for original_label, mapped_idx in label_mapping.items()
    }

    return index_to_description

class ImageNetValFlatDataset(Dataset):
    """
    ImageNet validation dataset for a flat image directory plus ground-truth txt.

    Expected structure:
        root/
            ILSVRC2012_devkit_t12/
                data/
                    ILSVRC2012_validation_ground_truth.txt
            ILSVRC2012_img_val/
                ILSVRC2012_val_00000001.JPEG
                ILSVRC2012_val_00000002.JPEG
                ...

    Notes:
    - Ground-truth labels in the txt are 1-based.
    - This implementation converts them to 0-based labels.
    """

    def __init__(
        self,
        root: str = "Data",
        transform: Callable = None,
    ) -> None:
        
        self.root = Path(root)
        self.transform = transform

        self.image_dir = self.root / "ILSVRC2012_img_val"
        self.gt_file = (
            self.root
            / "ILSVRC2012_devkit_t12"
            / "data"
            / "ILSVRC2012_validation_ground_truth.txt"
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.gt_file.exists():
            raise FileNotFoundError(f"Ground-truth file not found: {self.gt_file}")

        self.image_paths = sorted(self.image_dir.glob("*.JPEG"))
        if not self.image_paths:
            raise RuntimeError(f"No JPEG images found in {self.image_dir}")

        with open(self.gt_file, "r") as f:
            gt_labels = [int(line.strip()) for line in f if line.strip()]

        if len(self.image_paths) != len(gt_labels):
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) and labels ({len(gt_labels)}) do not match."
            )

        label_mapping = build_imagenet_label_mapping(f"{self.root}/ILSVRC2012_devkit_t12")
        # Convert 1-based ImageNet labels to 0-based class indices
        self.targets = [label_mapping[label] for label in gt_labels]
        self.class_num = max(self.targets) + 1
        self.info =  ImageNetInfo()
        self.index_to_description_map = build_index_to_description(f"{self.root}/ILSVRC2012_devkit_t12", label_mapping)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        target = self.targets[index]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def index_to_description(self, index: int) -> str:
        return self.index_to_description_map[index]


class UnifiedDataset(Dataset):
    def __init__(self, ds=None, transform=None, ds_name="imagenet-1k", split="validation"):
        self.ds = ds
        
        if self.ds == None and not ds_name=="imagenet-r":
            self.ds = load_dataset(ds_name, split=split)
        
        self.transform = transform
        if transform is None:
            raise ValueError("Not Allow Transform is None")
        

        # HF Dataset Check
        self.is_hf = hasattr(ds, "features")

        if self.is_hf:
            label_feature = ds.features["label"]
            self.class_num = label_feature.num_classes
            self.class_names = label_feature.names
        else:
            self.class_num = len(ds.classes)
            self.class_names = ds.classes
    
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.is_hf:
            item = self.ds[idx]
            image = item["image"]
            label = item["label"]
        else:
            image, label = self.ds[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
if __name__ == "__main__":
    from timm.data import resolve_model_data_config, create_transform
    
    def debug_label_mapping(devkit_dir: str, label_mapping: dict):
        devkit_dir = Path(devkit_dir)

        # 1) devkit label -> (wnid, description)
        meta = loadmat(
            str(devkit_dir / "data" / "meta.mat"),
            struct_as_record=False,
            squeeze_me=True,
        )
        synsets = meta["synsets"]

        original_info = {}
        for s in synsets:
            if int(s.num_children) == 0:
                original_info[int(s.ILSVRC2012_ID)] = {
                    "wnid": str(s.WNID),
                    "name": str(s.words),
                }

        # 2) timm index -> wnid
        raw = pkgutil.get_data("timm", "data/_info/imagenet_synsets.txt")
        timm_synsets = raw.decode("utf-8").strip().splitlines()

        # 3) 출력
        for k in range(1, 11):
            mapped_idx = label_mapping[k]
            wnid = original_info[k]["wnid"]
            name = original_info[k]["name"]

            print(
                f"{k} -> {mapped_idx} | {wnid} | {name} | timm_wnid={timm_synsets[mapped_idx]}"
            )

    label_mapping = build_imagenet_label_mapping("Data/ILSVRC2012_devkit_t12")
    debug_label_mapping("Data/ILSVRC2012_devkit_t12", label_mapping)
    
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit.eval()
    vit_config = resolve_model_data_config(vit)
    vit_transform = create_transform(**vit_config, is_training=False)

    print("="*15 + " ViT Config and Transform " + "="*15)
    print(vit_config)
    print(vit_transform)
    print("="*15  + "="*15 + "="*26)
    print()
    

    resnet = timm.create_model('resnet50', pretrained=True)
    resnet.eval()
    resnet_config = resolve_model_data_config(resnet)
    resnet_transform = create_transform(**resnet_config, is_training=False)
    
    print("="*15 + " ViT Config and Transform " + "="*15)
    print(resnet_config)
    print(resnet_transform)
    print("="*15  + "="*15 + "="*26)
    print()