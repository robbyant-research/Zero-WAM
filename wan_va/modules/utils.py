# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel
from .icl_model import WanICLTransformer3DModel


def resolve_model_component(model_root, component):
    component_path = Path(model_root).expanduser() / component
    if not component_path.is_dir():
        raise FileNotFoundError(
            f"Missing model component '{component}' under {model_root}"
        )
    return str(component_path)


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    disable_mcp=False,
    icl_model=False,
    **kwargs
):
    model_cls = WanICLTransformer3DModel if icl_model else WanTransformer3DModel
    torch_device = torch.device(torch_device)
    if torch_device.type != 'cpu':
        kwargs.setdefault('device_map', {'': str(torch_device)})
        kwargs.setdefault('low_cpu_mem_usage', True)
    model = model_cls.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    if disable_mcp and getattr(model, 'enable_mcp', False):
        model.disable_mcp_modules()
    # ``from_pretrained(torch_dtype=...)`` casts checkpoint parameters, but
    # buffers created by the model itself keep their constructor dtype.  Move
    # both parameters and non-persistent RoPE buffers to the inference dtype.
    return model.to(device=torch_device, dtype=torch_dtype)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
