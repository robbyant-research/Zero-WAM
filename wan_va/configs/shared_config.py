# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536
va_shared_cfg.infer_mode = 'server'
va_shared_cfg.device = 'auto'

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

_offload = os.getenv('ZERO_WAM_OFFLOAD', 'auto').lower()
if _offload not in {'auto', '0', '1', 'false', 'true'}:
	raise ValueError('ZERO_WAM_OFFLOAD must be auto, 0, 1, false, or true')
va_shared_cfg.enable_offload = (
	None if _offload == 'auto' else _offload in {'1', 'true'}
)

# Cache the validated latent sample index so repeated training runs do not scan
# every latent path on network storage again.
va_shared_cfg.enable_dataset_index_cache = True
va_shared_cfg.rebuild_dataset_index_cache = False
