import json
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from .dit.joint_model.context_parallel import get_cp_rank_size
from tqdm import tqdm
from comfy.utils import ProgressBar, load_torch_file

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

from .dit.joint_model.asymm_models_joint import AsymmDiTJoint

MAX_T5_TOKEN_LENGTH = 256

def unnormalize_latents(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Unnormalize latents. Useful for decoding DiT samples.

    Args:
        z (torch.Tensor): [B, C_z, T_z, H_z, W_z], float

    Returns:
        torch.Tensor: [B, C_z, T_z, H_z, W_z], float
    """
    mean = mean[:, None, None, None]
    std = std[:, None, None, None]

    assert z.ndim == 5
    assert z.size(1) == mean.size(0) == std.size(0)
    return z * std.to(z) + mean.to(z)

def compute_packed_indices(
    N: int,
    text_mask: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        N: Number of visual tokens.
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    assert N > 0 and len(text_mask) == 1
    text_mask = text_mask[0]

    mask = F.pad(text_mask, (N, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(
        mask.flatten(), as_tuple=False
    ).flatten()  # up to (B * (N + L),)

    assert valid_token_indices.size(0) >= text_mask.size(0) * N  # At least (B * N,)
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens,
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
    }


def shift_sigma(
    sigma: np.ndarray,
    shift: float,
):
    """Shift noise standard deviation toward higher values.

    Useful for training a model at high resolutions,
    or sampling more finely at high noise levels.

    Equivalent to:
        sigma_shift = shift / (shift + 1 / sigma - 1)
    except for sigma = 0.

    Args:
        sigma: noise standard deviation in [0, 1]
        shift: shift factor >= 1.
               For shift > 1, shifts sigma to higher values.
               For shift = 1, identity function.
    """
    return shift * sigma / (shift * sigma + 1 - sigma)


class T2VSynthMochiModel:
    def __init__(
        self,
        *,
        device: torch.device,
        offload_device: torch.device,
        vae_stats_path: str,
        dit_checkpoint_path: str,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
        fp8_fastmode: bool = False,
    ):
        super().__init__()
        self.device = device
        self.offload_device = offload_device

        print("Initializing model...")
        model: nn.Module = torch.nn.utils.skip_init(
            AsymmDiTJoint,
            depth=48,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
        )

        params_to_keep = {"t_embedder", "x_embedder", "pos_frequencies", "t5", "norm"}
        print(f"Loading model state_dict from {dit_checkpoint_path}...")
        dit_sd = load_torch_file(dit_checkpoint_path)
        if is_accelerate_available:
            print("Using accelerate to load and assign model weights to device...")
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    set_module_tensor_to_device(model, name, dtype=weight_dtype, device=self.device, value=dit_sd[name])
                else:
                    set_module_tensor_to_device(model, name, dtype=torch.bfloat16, device=self.device, value=dit_sd[name])
        else:
            print("Loading state_dict without accelerate...")
            model.load_state_dict(dit_sd)
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    param.data = param.data.to(weight_dtype)
                else:
                    param.data = param.data.to(torch.bfloat16)
        
        if fp8_fastmode:
            from ..fp8_optimization import convert_fp8_linear
            convert_fp8_linear(model, torch.bfloat16)

        self.dit = model
        self.dit.eval()
        
        vae_stats = json.load(open(vae_stats_path))
        self.vae_mean = torch.Tensor(vae_stats["mean"]).to(self.device)
        self.vae_std = torch.Tensor(vae_stats["std"]).to(self.device)

    def get_conditioning(self, prompts, *, zero_last_n_prompts: int):
        B = len(prompts)
        assert (
            0 <= zero_last_n_prompts <= B
        ), f"zero_last_n_prompts should be between 0 and {B}, got {zero_last_n_prompts}"
        tokenize_kwargs = dict(
            prompt=prompts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        t5_toks = self.t5_tokenizer(**tokenize_kwargs, max_length=MAX_T5_TOKEN_LENGTH)
        caption_input_ids_t5 = t5_toks["input_ids"]
        caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
        del t5_toks

        assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
        assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

        if zero_last_n_prompts > 0:
            # Zero the last N prompts
            caption_input_ids_t5[-zero_last_n_prompts:] = 0
            caption_attention_mask_t5[-zero_last_n_prompts:] = False

        caption_input_ids_t5 = caption_input_ids_t5.to(self.device, non_blocking=True)
        caption_attention_mask_t5 = caption_attention_mask_t5.to(
            self.device, non_blocking=True
        )

        y_mask = [caption_attention_mask_t5]
        y_feat = []

        self.t5_enc.to(self.device)
        y_feat.append(
            self.t5_enc(
                caption_input_ids_t5, caption_attention_mask_t5
            ).last_hidden_state.detach().to(torch.float32)
        )
        self.t5_enc.to(self.offload_device)
        # Sometimes returns a tensor, othertimes a tuple, not sure why
        # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
        assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
        return dict(y_mask=y_mask, y_feat=y_feat)

    def get_packed_indices(self, y_mask, *, lT, lW, lH):
        patch_size = 2
        N = lT * lH * lW // (patch_size**2)
        assert len(y_mask) == 1
        packed_indices = compute_packed_indices(N, y_mask)
        self.move_to_device_(packed_indices)
        return packed_indices

    def move_to_device_(self, sample):
        if isinstance(sample, dict):
            for key in sample.keys():
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(self.device, non_blocking=True)

    @torch.inference_mode(mode=True)
    def run(self, args, stream_results):
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

        generator = torch.Generator(device=self.device)
        generator.manual_seed(args["seed"])

        # assert (
        #     len(args["prompt"]) == 1
        # ), f"Expected exactly one prompt, got {len(args['prompt'])}"
        #prompt = args["prompt"][0]
        #neg_prompt = args["negative_prompt"][0] if len(args["negative_prompt"]) else ""
        B = 1

        w = args["width"]
        h = args["height"]
        t = args["num_frames"]
        batch_cfg = args["mochi_args"]["batch_cfg"]
        sample_steps = args["mochi_args"]["num_inference_steps"]
        cfg_schedule = args["mochi_args"].get("cfg_schedule")
        assert (
            len(cfg_schedule) == sample_steps
        ), f"cfg_schedule must have length {sample_steps}, got {len(cfg_schedule)}"
        sigma_schedule = args["mochi_args"].get("sigma_schedule")
        if sigma_schedule:
            assert (
                len(sigma_schedule) == sample_steps + 1
            ), f"sigma_schedule must have length {sample_steps + 1}, got {len(sigma_schedule)}"
        assert (t - 1) % 6 == 0, f"t - 1 must be divisible by 6, got {t - 1}"

        # if batch_cfg:
        #     sample_batched = self.get_conditioning(
        #         [prompt] + [neg_prompt], zero_last_n_prompts=B if neg_prompt == "" else 0
        #     )
        # else:
        #     sample = self.get_conditioning([prompt], zero_last_n_prompts=0)
        #     sample_null = self.get_conditioning([neg_prompt] * B, zero_last_n_prompts=B if neg_prompt == "" else 0)

        spatial_downsample = 8
        temporal_downsample = 6
        latent_t = (t - 1) // temporal_downsample + 1
        latent_w, latent_h = w // spatial_downsample, h // spatial_downsample

        latent_dims = dict(lT=latent_t, lW=latent_w, lH=latent_h)
        in_channels = 12
        z = torch.randn(
            (B, in_channels, latent_t, latent_h, latent_w),
            device=self.device,
            generator=generator,
            dtype=torch.float32,
        )

        # if batch_cfg:
        #     sample_batched["packed_indices"] = self.get_packed_indices(
        #         sample_batched["y_mask"], **latent_dims
        #     )
        #     z = repeat(z, "b ... -> (repeat b) ...", repeat=2)
        # else:

        sample = {
            "y_mask": [args["positive_embeds"]["attention_mask"].to(self.device)],
            "y_feat": [args["positive_embeds"]["embeds"].to(self.device)]
        }
        sample_null = {
            "y_mask": [args["negative_embeds"]["attention_mask"].to(self.device)],
            "y_feat": [args["negative_embeds"]["embeds"].to(self.device)]
        }

        # print(sample["y_mask"])
        # print(type(sample["y_mask"]))
        # print(sample["y_mask"][0].shape)

        # print(sample["y_feat"])
        # print(type(sample["y_feat"]))
        # print(sample["y_feat"][0].shape)

        # print(sample_null["y_mask"])
        # print(type(sample_null["y_mask"]))
        # print(sample_null["y_mask"][0].shape)

        # print(sample_null["y_feat"])
        # print(type(sample_null["y_feat"]))
        # print(sample_null["y_feat"][0].shape)

        sample["packed_indices"] = self.get_packed_indices(
            sample["y_mask"], **latent_dims
        )
        sample_null["packed_indices"] = self.get_packed_indices(
            sample_null["y_mask"], **latent_dims
        )

        def model_fn(*, z, sigma, cfg_scale):
            #print("z", z.dtype, z.device)
            #print("sigma", sigma.dtype, sigma.device)
            self.dit.to(self.device)
            # if batch_cfg:
            #     with torch.autocast("cuda", dtype=torch.bfloat16):
            #         out = self.dit(z, sigma, **sample_batched)
            #     out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
            #else:

            nonlocal sample, sample_null
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_cond = self.dit(z, sigma, **sample)
                out_uncond = self.dit(z, sigma, **sample_null)
            assert out_cond.shape == out_uncond.shape

            return out_uncond + cfg_scale * (out_cond - out_uncond), out_cond
        
        comfy_pbar = ProgressBar(sample_steps)
        for i in tqdm(range(0, sample_steps), desc="Processing Samples", total=sample_steps):
            sigma = sigma_schedule[i]
            dsigma = sigma - sigma_schedule[i + 1]

            # `pred` estimates `z_0 - eps`.
            pred, output_cond = model_fn(
                z=z,
                sigma=torch.full(
                    [B] if not batch_cfg else [B * 2], sigma, device=z.device
                ),
                cfg_scale=cfg_schedule[i],
            )
            pred = pred.to(z)
            output_cond = output_cond.to(z)

            #if stream_results:
            #    yield i / sample_steps, None, False
            z = z + dsigma * pred
            comfy_pbar.update(1)

        cp_rank, cp_size = get_cp_rank_size()
        if batch_cfg:
            z = z[:B]
        z = z.tensor_split(cp_size, dim=2)[cp_rank]  # split along temporal dim
        self.dit.to(self.offload_device)
    
        samples = unnormalize_latents(z.float(), self.vae_mean, self.vae_std)
        print("samples: ", samples.shape, samples.dtype, samples.device)
        return samples
