from torch.utils.data import Dataset
import cv2
import numpy
import albumentations as A
import timm


class UnifiedDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
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
    
class ImgNet1k(Dataset):
    def __init__(file_path, transform=None):
        pass
    
if __name__ == "__main__":
    from timm.data import resolve_model_data_config, create_transform

    
    
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