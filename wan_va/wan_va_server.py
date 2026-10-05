# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

from .configs import VA_CONFIGS
from .device import empty_cache, get_device
from .distributed.fsdp import shard_model
from .distributed.util import _configure_model, init_distributed
from .modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
    resolve_model_component,
)
from .modules.icl_model import (
    ICL_CACHE_TYPE,
    OBSERVATION_CACHE_TYPE,
    PREDICTION_CACHE_TYPE,
)
from .utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = get_device(job_config.local_rank, job_config.device)
        configured_offload = getattr(job_config, 'enable_offload', None)
        self.enable_offload = (
            self.device.type != 'cuda'
            if configured_offload is None
            else configured_offload
        )

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        model_root = job_config.model_path

        self.vae = load_vae(
            resolve_model_component(model_root, 'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            resolve_model_component(model_root, 'tokenizer'), )

        self.text_encoder = load_text_encoder(
            resolve_model_component(model_root, 'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        empty_text_emb_path = getattr(job_config, "empty_text_emb_path", "")
        self.empty_text_emb = None
        if empty_text_emb_path:
            empty_text_emb = torch.load(empty_text_emb_path, map_location="cpu")
            if not torch.is_tensor(empty_text_emb) or empty_text_emb.ndim != 2:
                raise ValueError(
                    "empty_text_emb_path must contain a [sequence, hidden] tensor"
                )
            self.empty_text_emb = empty_text_emb[None].to(
                device=self.device, dtype=self.dtype
            )

        self.use_icl_model = bool(getattr(job_config, 'use_icl_model', False))
        self.transformer = load_transformer(
            resolve_model_component(model_root, 'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            disable_mcp=not self.use_icl_model,
            icl_model=self.use_icl_model,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                resolve_model_component(model_root, 'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _encode_initial_obs(self, obs):
        """Encode the first Robotwin observation like the VA sync runtime."""
        image = obs['obs']
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    "The initial observation must contain exactly one frame"
                )
            image = image[0]

        self.streaming_vae.clear_cache()
        vae_device = next(self.streaming_vae.vae.parameters()).device
        videos = []
        for key in self.job_config.obs_cam_keys:
            current = torch.from_numpy(image[key].copy()).float().to(vae_device)
            current = current / 255.0 * 2.0 - 1.0
            current = current.permute(2, 0, 1).unsqueeze(0)
            current = F.interpolate(
                current,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False,
            )
            videos.append(current.unsqueeze(2))

        video = torch.cat(videos, dim=0).to(self.vae.dtype)
        encoded = self.streaming_vae.encode_chunk(video)
        mu, _ = torch.chunk(encoded, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return torch.cat(mu.split(1, dim=0), dim=-1).to(self.device)

    def _sample_icl_frame_indices(self, total_frames, src_fps, target_fps):
        if total_frames <= 0:
            raise ValueError("ICL video has no frames")
        if src_fps <= 0 or target_fps <= 0:
            return np.arange(total_frames, dtype=np.int64)
        output_frames = max(1, int(round(total_frames / src_fps * target_fps)))
        indices = np.floor(
            np.arange(output_frames, dtype=np.float64) * src_fps / target_fps
        ).astype(np.int64)
        return np.unique(np.clip(indices, 0, total_frames - 1))

    def _read_icl_video(self, video_path):
        target_h = int(self.job_config.icl_height)
        target_w = int(self.job_config.icl_width)
        target_fps = int(self.job_config.icl_fps)
        frames = None
        decord_error = None
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(video_path, ctx=cpu(0))
            indices = self._sample_icl_frame_indices(
                len(reader), float(reader.get_avg_fps() or 0), target_fps
            )
            frames = reader.get_batch(indices).asnumpy()
        except Exception as exc:
            decord_error = exc
        if frames is None:
            try:
                import imageio.v2 as imageio

                reader = imageio.get_reader(video_path)
                src_fps = float(reader.get_meta_data().get("fps", 0) or 0)
                frames = np.stack(list(reader), axis=0)
                reader.close()
                indices = self._sample_icl_frame_indices(
                    len(frames), src_fps, target_fps
                )
                frames = frames[indices]
            except Exception as exc:
                raise RuntimeError(f"Failed to read ICL video: {video_path}") from (
                    decord_error or exc
                )
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames = F.interpolate(
            frames,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def _reshape_icl_latent(self, latent, item):
        if not torch.is_tensor(latent):
            raise ValueError("ICL latent must be a tensor")
        layout = str(item.get("latent_layout", "")).lower().replace("_", " ")
        if latent.ndim == 5:
            if latent.shape[0] != 1:
                raise ValueError(f"ICL latent batch must be 1, got {latent.shape}")
            return latent.contiguous()
        if latent.ndim == 4:
            if layout in ("f h w c", "f h w d"):
                return latent.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            return latent.unsqueeze(0).contiguous()
        if latent.ndim == 2:
            f = int(item["latent_num_frames"])
            h = int(item["latent_height"])
            w = int(item["latent_width"])
            if latent.shape[0] != f * h * w:
                raise ValueError(
                    f"Flattened ICL latent {latent.shape} does not match {(f, h, w)}"
                )
            return latent.view(f, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0)
        raise ValueError(f"Unsupported ICL latent shape: {latent.shape}")

    def _load_or_encode_icl(self, video_path, latent_path):
        if latent_path and os.path.exists(latent_path):
            item = torch.load(latent_path, map_location="cpu")
            if not isinstance(item, dict) or "latent" not in item:
                raise ValueError(f"ICL latent file must contain 'latent': {latent_path}")
            latent = self._reshape_icl_latent(item["latent"], item)
            text_emb = item.get("text_emb")
            if torch.is_tensor(text_emb) and text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(0)
            logger.info(
                f"[ICL] loaded latent={tuple(latent.shape)} from {latent_path}"
            )
            return latent, text_emb
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(
                f"ICL input not found: latent={latent_path!r}, video={video_path!r}"
            )
        self.streaming_vae.clear_cache()
        video = self._read_icl_video(video_path).to(
            device=next(self.vae.parameters()).device, dtype=self.dtype
        )
        video = video / 255.0 * 2.0 - 1.0
        enc_out = self.streaming_vae.encode_chunk(video)
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean, device=mu.device)
        std = torch.tensor(self.vae.config.latents_std, device=mu.device)
        latent = self.normalize_latents(mu, mean, 1.0 / std).cpu()
        self.streaming_vae.clear_cache()
        logger.info(f"[ICL] encoded latent={tuple(latent.shape)} from {video_path}")
        return latent, None

    def _cache_icl_context(self, video_path, latent_path):
        latent, text_emb = self._load_or_encode_icl(video_path, latent_path)
        expected_channels = len(self.vae.config.latents_mean)
        if latent.shape[1] != expected_channels:
            raise ValueError(
                f"ICL latent has {latent.shape[1]} channels; expected {expected_channels}"
            )
        text_emb = self.prompt_embeds if text_emb is None else text_emb.to(self.device)
        latent = latent.to(device=self.device, dtype=self.dtype)
        patch_size = self.transformer.patch_size
        grid_id = get_mesh_id(
            latent.shape[-3] // patch_size[0],
            latent.shape[-2] // patch_size[1],
            latent.shape[-1] // patch_size[2],
            0,
            1,
            0,
        ).to(self.device)
        grid_id[1] += int(self.job_config.icl_rope_h)
        token_count = grid_id.shape[1]
        input_dict = {
            "latent_res_lst": {
                "noisy_latents": latent,
                "timesteps": torch.zeros(
                    latent.shape[2], device=self.device, dtype=torch.float32
                ),
                "cache_type_ids": torch.full(
                    [token_count], ICL_CACHE_TYPE, device=self.device, dtype=torch.int
                ),
            },
            "latent_grid_id": grid_id,
            "current_seq_ids": torch.zeros(
                token_count, device=self.device, dtype=torch.int
            ),
            # Bidirectional ICL is cached as one context block at frame id 0.
            # Temporal RoPE still comes from latent_grid_id.
            "current_frame_ids": torch.zeros(
                token_count, device=self.device, dtype=torch.int
            ),
            "encoder_seq_ids": torch.zeros(
                text_emb.shape[1], device=self.device, dtype=torch.int
            ),
            "text_emb": text_emb.to(device=self.device, dtype=self.dtype),
        }
        self.transformer(
            input_dict,
            update_cache=1,
            cache_name=self.cache_name,
            mode="forward_latent_only",
        )
        logger.info(f"[ICL] persistent cache: {self.transformer.cache_counts()}")

    def _reset_icl(
        self,
        prompt,
        use_icl,
        icl_video_path,
        icl_latent_path,
        video_guidance_scale,
        icl_guidance_scale,
    ):
        if not prompt:
            raise ValueError("Robotwin ICL inference requires a prompt")
        self.chunk_idx = 0
        self.init_latent = None
        self.last_predicted_latents = None
        self.last_predicted_actions = None
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width
        if self.env_type == 'robotwin_tshape':
            self.latent_height = ((self.height // 16) * 3) // 2
            self.latent_width = self.width // 16
        else:
            self.latent_height = self.height // 16
            self.latent_width = (
                self.width // 16 * len(self.job_config.obs_cam_keys)
            )
        self.action_mask = torch.zeros(
            self.job_config.action_dim, dtype=torch.bool, device=self.device
        )
        self.action_mask[self.job_config.used_action_channel_ids] = True
        self.actions_q01 = torch.tensor(
            self.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(
            self.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=self.device,
            dtype=self.dtype,
        )
        if self.empty_text_emb is not None:
            self.negative_prompt_embeds = self.empty_text_emb.clone()
        self.use_icl = bool(use_icl)
        self.video_guidance_scale = float(video_guidance_scale)
        self.icl_guidance_scale = float(icl_guidance_scale)
        self.use_icl_cfg = self.use_icl and self.icl_guidance_scale > 1.0
        self.target_text_cfg_active = (
            not self.use_icl_cfg and self.video_guidance_scale > 1.0
        )
        self.target_prompt_embeds = (
            self.negative_prompt_embeds
            if self.video_guidance_scale < 0
            else self.prompt_embeds
        )
        if self.use_icl:
            self._cache_icl_context(icl_video_path, icl_latent_path)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()

    def _pack_icl_cfg(self, input_dict, mode, cache_type):
        stream_key = "latent_res_lst" if mode == "video" else "action_res_lst"
        grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
        stream = input_dict[stream_key]
        grid_id = input_dict[grid_key]
        base_tokens = grid_id.shape[1]
        cfg_active = self.use_icl_cfg if mode == "video" else False
        if mode == "video" and self.target_text_cfg_active:
            cfg_active = True

        if cfg_active:
            stream["noisy_latents"] = stream["noisy_latents"].repeat(
                1, 1, 2, 1, 1
            )
            stream["timesteps"] = stream["timesteps"].repeat(2)
            input_dict[grid_key] = grid_id.repeat(1, 2)
            input_dict["current_seq_ids"] = torch.cat(
                [
                    torch.zeros(base_tokens, device=self.device, dtype=torch.int),
                    torch.ones(base_tokens, device=self.device, dtype=torch.int),
                ]
            )
            second_text = (
                self.target_prompt_embeds
                if self.use_icl_cfg
                else self.negative_prompt_embeds
            )
            input_dict["text_emb"] = torch.cat(
                [self.target_prompt_embeds, second_text], dim=1
            ).to(self.dtype)
            text_tokens = self.target_prompt_embeds.shape[1]
            input_dict["encoder_seq_ids"] = torch.cat(
                [
                    torch.zeros(text_tokens, device=self.device, dtype=torch.int),
                    torch.ones(text_tokens, device=self.device, dtype=torch.int),
                ]
            )
        else:
            input_dict["current_seq_ids"] = torch.zeros(
                base_tokens, device=self.device, dtype=torch.int
            )
            input_dict["text_emb"] = self.target_prompt_embeds.to(self.dtype)
            input_dict["encoder_seq_ids"] = torch.zeros(
                self.target_prompt_embeds.shape[1],
                device=self.device,
                dtype=torch.int,
            )
        token_count = input_dict[grid_key].shape[1]
        frame_id = 2 * self.chunk_idx + (1 if mode == "action" else 0)
        input_dict["current_frame_ids"] = torch.full(
            [token_count], frame_id, device=self.device, dtype=torch.int
        )
        stream["cache_type_ids"] = torch.full(
            [token_count], cache_type, device=self.device, dtype=torch.int
        )
        return cfg_active

    def _prepare_icl_stream(self, data, timestep, mode, cache_type):
        grid_id = get_mesh_id(
            data.shape[-3],
            data.shape[-2]
            // (self.transformer.patch_size[1] if mode == "video" else 1),
            data.shape[-1]
            // (self.transformer.patch_size[2] if mode == "video" else 1),
            0 if mode == "video" else 1,
            1,
            self.job_config.frame_chunk_size * self.chunk_idx,
            action=False,
        ).to(self.device)
        stream_key = "latent_res_lst" if mode == "video" else "action_res_lst"
        grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
        input_dict = {
            stream_key: {
                "noisy_latents": data,
                "timesteps": torch.full(
                    [data.shape[2]],
                    float(timestep),
                    device=self.device,
                    dtype=torch.float32,
                ),
            },
            grid_key: grid_id,
        }
        self._pack_icl_cfg(input_dict, mode, cache_type)
        return input_dict

    def _infer_icl(self, obs):
        if self.chunk_idx == 0:
            self.init_latent = self._encode_initial_obs(obs)
        latents = torch.randn(
            1,
            48,
            self.job_config.frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            self.job_config.frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        # Consume inactive-channel noise before masking so seeded sampling stays
        # deterministic across chunks.
        _ = torch.randn_like(actions)[:, ~self.action_mask]
        self.scheduler.set_timesteps(self.job_config.num_inference_steps)
        self.action_scheduler.set_timesteps(
            self.job_config.action_num_inference_steps
        )
        video_timesteps = torch.cat(
            [
                self.scheduler.timesteps,
                torch.zeros(
                    1,
                    device=self.scheduler.timesteps.device,
                    dtype=self.scheduler.timesteps.dtype,
                ),
            ]
        )

        for step, timestep in enumerate(video_timesteps):
            last_step = step == len(video_timesteps) - 1
            latent_input = latents.clone()
            if self.chunk_idx == 0:
                latent_input[:, :, :1] = self.init_latent[:, :, :1]
            input_dict = self._prepare_icl_stream(
                latent_input, timestep, "video", PREDICTION_CACHE_TYPE
            )
            if self.chunk_idx == 0:
                # CFG repeats the temporal axis as [cond frames, uncond
                # frames].  The observed first frame is clean in every branch.
                input_dict["latent_res_lst"]["timesteps"][
                    ::self.job_config.frame_chunk_size
                ] = 0
            noise = self.transformer(
                input_dict,
                update_cache=int(last_step),
                cache_name=self.cache_name,
                mode="forward_latent_only",
            )
            if not last_step:
                if self.use_icl_cfg:
                    noise_cond, noise_uncond = noise.chunk(2, dim=1)
                    noise = noise_uncond + self.icl_guidance_scale * (
                        noise_cond - noise_uncond
                    )
                elif self.target_text_cfg_active:
                    noise_cond, noise_uncond = noise.chunk(2, dim=1)
                    noise = noise_uncond + self.video_guidance_scale * (
                        noise_cond - noise_uncond
                    )
                noise = data_seq_to_patch(
                    self.transformer.patch_size,
                    noise,
                    self.job_config.frame_chunk_size,
                    self.latent_height,
                    self.latent_width,
                )
                latents = self.scheduler.step(
                    noise, timestep, latents, return_dict=False
                )
                if self.chunk_idx == 0:
                    latents[:, :, :1] = self.init_latent[:, :, :1]

        for timestep in self.action_scheduler.timesteps:
            action_input = actions.clone()
            action_input[:, ~self.action_mask] = 0
            input_dict = self._prepare_icl_stream(
                action_input, timestep, "action", PREDICTION_CACHE_TYPE
            )
            action_noise = self.transformer(
                input_dict,
                update_cache=0,
                cache_name=self.cache_name,
                mode="forward_action_only",
            )
            action_noise = rearrange(
                action_noise,
                "1 (f n) c -> 1 c f n 1",
                f=self.job_config.frame_chunk_size,
            )
            actions = self.action_scheduler.step(
                action_noise, timestep, actions, return_dict=False
            )
        actions[:, ~self.action_mask] = 0
        cached_actions = actions.clone()
        if self.chunk_idx == 0:
            cached_actions[:, :, 0] = 0
        self.last_predicted_latents = latents
        self.last_predicted_actions = cached_actions
        logger.info(f"[ICL] cache after infer: {self.transformer.cache_counts()}")
        return self.postprocess_action(actions)

    def _compute_icl_kv_cache(self, obs):
        # Remove the generated video chunk before replacing it with encoded
        # observations. Window pruning happens after the matching action cache
        # is appended.
        self.transformer.clear_prediction_cache(
            -1, self.cache_name
        )
        latent_input = self._encode_obs(obs)
        if self.chunk_idx == 0:
            latent_input = torch.cat([self.init_latent, latent_input], dim=2)
        if self.last_predicted_actions is None:
            raise RuntimeError("Action history requested before the first ICL inference")
        # Cache normalized model output directly to avoid a postprocess/preprocess
        # round trip.
        action_input = self.last_predicted_actions.to(latent_input)

        video_input = self._prepare_icl_stream(
            latent_input, 0, "video", OBSERVATION_CACHE_TYPE
        )
        self.transformer(
            video_input,
            update_cache=1,
            cache_name=self.cache_name,
            mode="forward_latent_only",
        )
        action_dict = self._prepare_icl_stream(
            action_input, 0, "action", OBSERVATION_CACHE_TYPE
        )
        self.transformer(
            action_dict,
            update_cache=1,
            cache_name=self.cache_name,
            mode="forward_action_only",
            clean_window_cache=True,
        )
        self.chunk_idx += 1
        logger.info(f"[ICL] cache after observation: {self.transformer.cache_counts()}")

    def _reset(
        self,
        prompt=None,
        use_icl=False,
        icl_video_path="",
        icl_latent_path="",
        video_guidance_scale=None,
        icl_guidance_scale=None,
    ):
        if self.use_icl_model:
            return self._reset_icl(
                prompt,
                use_icl,
                icl_video_path,
                icl_latent_path,
                self.job_config.guidance_scale
                if video_guidance_scale is None
                else video_guidance_scale,
                self.job_config.icl_guidance_scale
                if icl_guidance_scale is None
                else icl_guidance_scale,
            )
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        empty_cache(self.device)

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        empty_cache(self.device)
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        empty_cache(self.device)
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if self.use_icl_model:
            if reset:
                self._reset(
                    prompt=prompt,
                    use_icl=bool(obs.get("use_icl", True)),
                    icl_video_path=obs.get("icl_video_path", ""),
                    icl_latent_path=obs.get("icl_latent_path", ""),
                    video_guidance_scale=obs.get(
                        "video_guidance_scale", self.job_config.guidance_scale
                    ),
                    icl_guidance_scale=obs.get(
                        "icl_guidance_scale", self.job_config.icl_guidance_scale
                    ),
                )
                return dict()
            if compute_kv_cache:
                self._compute_icl_kv_cache(obs)
                return dict()
            return dict(action=self._infer_icl(obs))

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        empty_cache(self.device)
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank, config.device)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info("******************************USE I2VA mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
