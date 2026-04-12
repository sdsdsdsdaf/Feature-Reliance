from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

import albumentations as A

from torchvision import transforms
import os, platform, sys, psutil, torch

def get_system_info():
    print("===== System Info =====")
    
    # OS
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Python
    print(f"Python Version: {sys.version.split()[0]}")
    
    # CPU
    print(f"CPU: {platform.processor()}")
    
    # RAM
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM: {ram:.2f} GB")
    
    print("\n===== PyTorch Info =====")
    
    # PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    
    # GPU + VRAM
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            print(f"GPU {i}: {props.name}, {vram:.2f} GB VRAM")
    else:
        print("GPU: Not available")

def load_imagenet(normalize=True):
    pass