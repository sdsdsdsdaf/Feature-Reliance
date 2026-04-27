from torch.utils.data import Dataset
import cv2
import numpy
import albumentations as A
import timm
from pathlib import Path
from typing import Callable, Optional, List, Sequence

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from scipy.io import loadmat
import pkgutil
from timm.data import ImageNetInfo
from torchvision.datasets import ImageFolder

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

class ImageNetValSubsetDataset(Dataset):
    def __init__(
        self,
        indices: Sequence[int],
        class_ids: Sequence[int],
        root: Optional[str] = None,
        transform=None,
    ):
        self.base_ds = (
            ImageNetValFlatDataset(root=root, transform=transform)
            if root is not None
            else ImageNetValFlatDataset(transform=transform)
        )
        self.indices = list(indices)
        self.class_ids = list(class_ids)
        self.class_id_to_subset_idx = {
            class_id: i for i, class_id in enumerate(self.class_ids)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_ds[self.indices[idx]]
        label = self.class_id_to_subset_idx[label]
        return image, label


def build_sample_indices_from_targets(
    targets: Sequence[int],
    class_ids: Sequence[int],
) -> List[int]:
    class_id_set = set(int(x) for x in class_ids)
    return [i for i, y in enumerate(targets) if int(y) in class_id_set]

class ImageFolderDS(Dataset):
    
    def __init__(self, root:str, transform=None):
        super().__init__()
        
        self.ds = ImageFolder(root)
        self.transform = transform
        self.root = root
        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, label = self.ds[index]
        arr = np.array(image)
        
        if self.transform is not None:
            arr = self.transform(arr)
            
        return arr, label


class HFImageNetTrainSubsetDataset(Dataset):
    def __init__(self, hf_dataset, class_ids, clean_transform=None, perturb_transform=None):
        self.ds = hf_dataset
        self.class_ids = set(int(x) for x in class_ids)
        self.clean_transform = clean_transform
        self.perturb_transform = perturb_transform

        labels = self.ds["label"]
        self.indices = [
            i for i, y in enumerate(labels)
            if int(y) in self.class_ids
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.ds[self.indices[idx]]

        image = item["image"].convert("RGB")
        image = np.array(image)
        label = int(item["label"])  # keep 0~999 label

        if self.perturb_transform is None:
            if self.clean_transform is not None:
                image = self.clean_transform(image)
            return image, label

        clean = self.clean_transform(image)
        perturbed = self.perturb_transform(image)

        return {
            "clean": clean,
            "perturbed": perturbed,
            "label": label,
        }
    
if __name__ == "__main__":
    from timm.data import resolve_model_data_config, create_transform
    
    """
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
    """
    
    from datasets import load_from_disk
    from utils import CLASS_MAPPING_REGISTRY, build_transform
    from Config import ModelSpec, DataConfig, TransformHyperParams
    
    imagenet_r_class_ids = CLASS_MAPPING_REGISTRY["imagenet_r_subset_map"]["subset_class_ids"]
    
    ds = load_from_disk("Data/ILSVRC2012/train")
    
    print(type(ds))
    print(ds.column_names)
    
    model_spec = ModelSpec(
        model_name="resnet50",
        pretrained_weight="in1k",
        model=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_size=235, # ResNet50의 경우 256으로 resize 후 center crop 224 진행 (timm의 기본 전처리 방식)
    )
    
    clean_transform = build_transform(
        perturbation="original",
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=TransformHyperParams(
            resize_size=256,
            p=1.0,
            prefix="resizecrop",
            bilateral_d=11,
            sigma_color=170,
            sigma_space=75,
            gaussian_k=11,
            gaussian_sigma=2.0,
            gray_alpha=1.0,
         ),
        normalize=True,
    )
    
    perturb_transform = build_transform(
        perturbation="localwarp",
        mean=model_spec.mean,
        std=model_spec.std,
        resize_size=model_spec.resize_size,
        hparams=TransformHyperParams(
        resize_size=256,
        p=1.0,
        prefix="resizecrop",
        bilateral_d=11,
        sigma_color=170,
        sigma_space=75,
        gaussian_k=11,
        gaussian_sigma=2.0,
        gray_alpha=1.0,
        ),
        normalize=True,
    )
    
    train_subset = HFImageNetTrainSubsetDataset(
        hf_dataset=ds,
        class_ids=imagenet_r_class_ids,   # 0~999 ImageNet-R aligned class ids
        clean_transform=clean_transform,
        perturb_transform=perturb_transform,
    )
    
    from tqdm.auto import tqdm
    import torch
    
    def sanity_check_intervention_dataset(train_subset, n=5):
        print("dataset length:", len(train_subset))

        labels = []

        for i in range(n):
            item = train_subset[i]

            clean = item["clean"]
            pert = item["perturbed"]
            label = item["label"]

            labels.append(int(label))

            print(f"\n[{i}]")
            print("label:", label)
            print("clean type/shape:", type(clean), clean.shape, clean.dtype)
            print("pert  type/shape:", type(pert), pert.shape, pert.dtype)

            if torch.is_tensor(clean):
                print("clean min/max/mean:", clean.min().item(), clean.max().item(), clean.mean().item())
                print("pert  min/max/mean:", pert.min().item(), pert.max().item(), pert.mean().item())
            print("abs diff mean:", (clean - pert).abs().mean().item())

        print("\nunique labels in sampled items:", sorted(set(labels)))
        
        

    sanity_check_intervention_dataset(train_subset, n=10)
    from torch.utils.data import DataLoader

    loader = DataLoader(train_subset, batch_size=512, shuffle=False)

    labels = np.array(train_subset.ds["label"])
    subset_indices = np.array(train_subset.indices)

    subset_labels = labels[subset_indices]

    unique_labels = set(subset_labels.tolist())

    print("num unique:", len(unique_labels))

    expected = set(CLASS_MAPPING_REGISTRY["imagenet_r_subset_map"]["subset_class_ids"])
    actual = unique_labels

    print("match:", actual == expected)
    print("missing:", expected - actual)
    print("extra:", actual - expected)