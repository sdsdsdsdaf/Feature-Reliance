import torch
import timm
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

def normalize_resnet_layers(target_layers):
    # 단일 값 → 리스트화
    if isinstance(target_layers, (str, int)):
        target_layers = [target_layers]

    valid_layers = ["layer1", "layer2", "layer3", "layer4"]
    normalized = []

    for layer_name in target_layers:

        # --- case 1: "lastk" ---
        if isinstance(layer_name, str) and layer_name.startswith("last"):
            k = int(layer_name.replace("last", ""))
            if not (1 <= k <= 4):
                raise ValueError(f"Invalid last{k} for ResNet")

            normalized.extend(valid_layers[-k:])
            continue

        # --- case 2: int index ---
        if isinstance(layer_name, int):
            if 1 <= layer_name <= 4:
                layer_name = f"layer{layer_name}"
            elif -4 <= layer_name <= -1:
                layer_name = f"layer{4 + layer_name + 1}"
            else:
                raise ValueError(f"Invalid layer index: {layer_name}")

        # --- case 3: string layer ---
        if layer_name not in valid_layers:
            raise ValueError(
                f"Invalid layer name: {layer_name}. "
                f"Must be one of {valid_layers}"
            )

        normalized.append(layer_name)

    # 중복 제거 + 순서 유지
    normalized = list(dict.fromkeys(normalized))

    return normalized

class LinearAdaptor(nn.Module):
    def __init__(
        self,
        dim,
        reduction=16,
        use_norm=False,
        use_trainable_scale=False,
        init_scale=1e-3,
        dropout=0.0,
        safe_fp32_delta=True,
    ):
        super().__init__()
        hidden = max(dim // reduction, 1)

        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)

        self.safe_fp32_delta = safe_fp32_delta

        if use_trainable_scale:
            self.scale = nn.Parameter(
                torch.tensor(float(init_scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "scale",
                torch.tensor(float(init_scale), dtype=torch.float32),
            )

        self.last_delta = None
        self.last_delta_ratio = None

        # Important: start as near-identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.norm(x)
        z = self.down(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.up(z)

        if self.safe_fp32_delta:
            adapted = (self.scale.float() * z.float()).to(dtype=x.dtype)
        else:
            adapted = self.scale * z

        if self.training:
            # Keep graph for delta regularization.
            self.last_delta = adapted

            # Logging only. No graph.
            with torch.no_grad():
                adapted_f = adapted.detach().float().reshape(adapted.shape[0], -1)
                x_f = x.detach().float().reshape(x.shape[0], -1)

                self.last_delta_ratio = (
                    adapted_f.norm(dim=1).mean()
                    / (x_f.norm(dim=1).mean() + 1e-8)
                )

        return x + adapted
    
class ConvAdaptor(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
        use_trainable_scale=True,
        init_scale=1e-3,
        dropout=0.0,
        safe_fp32_delta=True,
    ):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.down = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1)

        self.safe_fp32_delta = safe_fp32_delta

        if use_trainable_scale:
            self.scale = nn.Parameter(
                torch.tensor(float(init_scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "scale",
                torch.tensor(float(init_scale), dtype=torch.float32),
            )

        self.last_delta = None
        self.last_delta_ratio = None

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.drop(z)
        z = self.up(z)

        if self.safe_fp32_delta:
            adapted = (self.scale.float() * z.float()).to(dtype=x.dtype)
        else:
            adapted = self.scale * z

        if self.training:
            # Keep graph for delta regularization.
            self.last_delta = adapted

            # Logging only. No graph.
            with torch.no_grad():
                adapted_f = adapted.detach().float().reshape(adapted.shape[0], -1)
                x_f = x.detach().float().reshape(x.shape[0], -1)

                self.last_delta_ratio = (
                    adapted_f.norm(dim=1).mean()
                    / (x_f.norm(dim=1).mean() + 1e-8)
                )

        return x + adapted
    
class ConvBlockWithAdaptor(nn.Module):
    def __init__(self, block, channels, **adaptor_kwargs):
        super().__init__()
        self.block = block
        self.adaptor = ConvAdaptor(channels=channels, **adaptor_kwargs)

    def forward(self, x):
        out = self.block(x)
        out = self.adaptor(out)
        return out
    
class TimmBlockWithAdaptor(nn.Module):
    def __init__(self, block, dim, adaptor_kwargs):
        super().__init__()
        self.block = block
        self.adaptor = LinearAdaptor(dim=dim, **adaptor_kwargs)

    def forward(self, x):
        x = self.block(x)
        x = self.adaptor(x)
        return x
    
class HFDinoLayerWithAdaptor(nn.Module):
    def __init__(self, layer, dim, adaptor_kwargs):
        super().__init__()
        self.layer = layer
        self.adaptor = LinearAdaptor(dim=dim, **adaptor_kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.layer(hidden_states, *args, **kwargs)

        # case 1: tuple output
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            hidden_states = self.adaptor(hidden_states)
            return (hidden_states,) + outputs[1:]

        # case 2: tensor output
        hidden_states = self.adaptor(outputs)
        return hidden_states
    
def inject_resnet_adaptors(
    model,
    target_layers=("layer4",),
    reduction=16,
    use_trainable_scale=False,
    use_norm=False,
    init_scale=1e-3,
    dropout=0.0,
):
    channels = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    }
    
    target_layers = normalize_resnet_layers(target_layers)
            
    for layer_name in target_layers:
        if layer_name not in channels:
            raise ValueError(f"Invalid layer name: {layer_name}. Must be one of {list(channels.keys())}")

    for layer_name in target_layers:
        old_layer = getattr(model, layer_name)

        setattr(
            model, 
            layer_name, 
            ConvBlockWithAdaptor(
                old_layer, 
                channels[layer_name], 
                reduction=reduction, 
                use_trainable_scale=use_trainable_scale,
                init_scale=init_scale,
                dropout=dropout,
            )
        )

    return model

def inject_timm_vit_adaptors(
    model:nn.Module,
    target_blocks="last4",
    reduction=16,
    use_norm=False,
    use_trainable_scale=False,
    init_scale=1e-3,
    dropout=0.0,
):
    num_blocks = len(model.blocks)
    dim = model.embed_dim
    
    if target_blocks == "last4":
        indices = list(range(num_blocks-4, num_blocks))
    elif target_blocks == "last2":
        indices = list(range(num_blocks-2, num_blocks))
    elif target_blocks == "last1":
        indices = [num_blocks-1]
    elif target_blocks == "all":
        indices = list(range(num_blocks))
    else:
        indices = target_blocks
        
    adaptor_kwargs = {
        "reduction": reduction,
        "use_norm": use_norm,
        "use_trainable_scale": use_trainable_scale,
        "init_scale": init_scale,
        "dropout": dropout,
    }
    
    for idx in indices:
        old_block = model.blocks[idx]
        model.blocks[idx] = TimmBlockWithAdaptor(old_block, dim, adaptor_kwargs=adaptor_kwargs)

    return model

def inject_hf_dino_adaptors(
    model:nn.Module,
    target_layers="last4",
    reduction=16,
    use_norm=False,
    use_trainable_scale=False,
    init_scale=1e-3,
    dropout=0.0,
):
    num_layers = len(model.dinov2.encoder.layer)
    dim = model.config.hidden_size
    
    if target_layers == "last4":
        indices = list(range(num_layers-4, num_layers))
    elif target_layers == "last2":
        indices = list(range(num_layers-2, num_layers))
    elif target_layers == "last1":
        indices = [num_layers-1]
    elif target_layers == "all":
        indices = list(range(num_layers))
    else:
        indices = target_layers
        
    adaptor_kwargs = {
        "reduction": reduction,
        "use_norm": use_norm,
        "use_trainable_scale": use_trainable_scale,
        "init_scale": init_scale,
        "dropout": dropout,
    }
    
    for idx in indices:
        old_layer = model.dinov2.encoder.layer[idx]
        model.dinov2.encoder.layer[idx] = HFDinoLayerWithAdaptor(old_layer, dim, adaptor_kwargs=adaptor_kwargs)

    return model

def inject_adaptors(
    model: nn.Module,
    model_type: str,
    target="last4",
    reduction=16,
    use_norm=False,
    use_trainable_scale=False,
    init_scale=1e-3,
    dropout=0.0,
):
    model_type = model_type.lower()

    if model_type in ["resnet", "resnet50", "cnn"]:
        return inject_resnet_adaptors(
            model=model,
            target_layers=target if isinstance(target, (list, tuple)) else (target,),
            reduction=reduction,
            use_trainable_scale=use_trainable_scale,
            init_scale=init_scale,
            dropout=dropout,
        )
        
    elif model_type in ["hf_dino", "dinov2", "dinov2_b14", "dino"]:
        return inject_hf_dino_adaptors(
            model=model,
            target_layers=target,
            reduction=reduction,
            use_norm=use_norm,
            use_trainable_scale=use_trainable_scale,
            init_scale=init_scale,
            dropout=dropout,
        )

    elif model_type in ["timm_vit", "vit", "vit_b16", "vit-b/16", "vit-b"]:
        return inject_timm_vit_adaptors(
            model=model,
            target_blocks=target,
            reduction=reduction,
            use_norm=use_norm,
            use_trainable_scale=use_trainable_scale,
            init_scale=init_scale,
            dropout=dropout,
        )

    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Use one of: resnet50, timm_vit, dinov2"
        ) 
 
if __name__ == "__main__":
    resnet = timm.create_model("resnet50", pretrained=True)
    vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    dinov2 = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-base-imagenet1k-1-layer",
        use_safetensors=True
    )
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    resnet = inject_adaptors(resnet, model_type="resnet", target="last2")
    vit = inject_adaptors(vit, model_type="timm_vit", target="last2")
    dinov2 = inject_adaptors(dinov2, model_type="hf_dino", target="last2")
    
    resnet.eval()
    vit.eval()
    dinov2.eval()
    
    resnet_out = resnet(dummy_input)
    vit_out = vit(dummy_input)
    dinov2_out = dinov2(dummy_input)
    
    # Structure test
    print()
    print("="*20 + " ResNet with Adaptors: " + "="*20)
    print(resnet)
    print("\n" + "="*20 + " ViT with Adaptors: " + "="*20)
    print(vit)
    print("\n" + "="*20 + " DINOv2 with Adaptors: " + "="*20)
    print(dinov2)
    
    # Forward pass test
    print("\n" + "="*20 + " Forward Pass Test: " + "="*20)
    resnet_adaptor_out = resnet(dummy_input)
    vit_adaptor_out = vit(dummy_input)
    dinov2_adaptor_out = dinov2(dummy_input)
    
    print("\nForward pass successful!")
    print(f"ResNet output shape: {resnet_adaptor_out.shape}")
    print(f"ViT output shape: {vit_adaptor_out.shape}")
    print(f"DINOv2 output: {dinov2_adaptor_out.__dict__.keys()}")
    print(f"DINOv2 logits shape: {dinov2_adaptor_out.logits.shape}") 
    
    # All close to zero due to near-identity initialization
    print("\nAdaptor output magnitudes (should be small):")
    print(f"ResNet adaptor output mean abs: {torch.mean(torch.abs(resnet_adaptor_out - resnet_out)).item():.6f}")
    print(f"ViT adaptor output mean abs: {torch.mean(torch.abs(vit_adaptor_out - vit_out)).item():.6f}")
    print(f"DINOv2 adaptor output mean abs: {torch.mean(torch.abs(dinov2_adaptor_out.logits - dinov2_out.logits)).item():.6f}")
