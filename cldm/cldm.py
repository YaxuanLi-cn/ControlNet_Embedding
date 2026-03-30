import math

import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class HeadingSinusoidalEncoder(nn.Module):
    """Encode heading angle (degrees) into paired sinusoidal features.

    Output: [cos(1·x), sin(1·x), cos(2·x), sin(2·x), …, cos(K·x), sin(K·x)]
    where x = heading in radians.  Output dim = 2K.
    Frequencies 1..K are fixed (not learnable).
    """

    def __init__(self, K: int = 4):
        super().__init__()
        self.K = K
        # Fixed integer frequencies 1, 2, …, K
        freqs = torch.arange(1, K + 1, dtype=torch.float32)  # (K,)
        self.register_buffer('freqs', freqs)

    def forward(self, heading_deg: torch.Tensor) -> torch.Tensor:
        """Args:
            heading_deg: (B,) heading values in degrees.
        Returns:
            (B, 2K) sinusoidal encoding.
        """
        x_rad = heading_deg.float() * (math.pi / 180.0)          # (B,)
        angles = x_rad.unsqueeze(-1) * self.freqs.unsqueeze(0)    # (B, K)
        # Interleave cos/sin: [cos(1x), sin(1x), cos(2x), sin(2x), ...]
        enc = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, K, 2)
        return enc.reshape(heading_deg.shape[0], 2 * self.K)      # (B, 2K)


class RangeRBFEncoder(nn.Module):
    """Encode range/distance with normalisation + RBF basis.

    Output: [d_norm, rbf_1(d_norm), rbf_2(d_norm), …, rbf_{2K-1}(d_norm)]
    where  rbf_k(d) = exp( -(d - c_k)^2 / (2·sigma^2) )
    centres are fixed & uniformly spaced on [0, 1].
    sigma is a shared learnable parameter (always > 0 via softplus).
    Output dim = 1 + (2K - 1) = 2K.
    """

    def __init__(self, K: int = 4, range_max: float = 132.0, range_min: float = -132.0):
        super().__init__()
        self.K = K
        self.range_max = range_max
        self.range_min = range_min
        num_rbf = 2 * K - 1
        centers = torch.linspace(0.0, 1.0, num_rbf)  # (num_rbf,)
        self.register_buffer('centers', centers)
        # Learnable raw parameter; sigma = softplus(raw_sigma) + eps
        self.raw_sigma = nn.Parameter(torch.tensor(0.0))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 1e-6

    def forward(self, range_num: torch.Tensor) -> torch.Tensor:
        """Args:
            range_num: (B,) raw distance values.
        Returns:
            (B, 2K) distance encoding.
        """
        # Normalise to [0, 1] and clamp
        d_norm = (range_num.float() - self.range_min) / (self.range_max - self.range_min)
        d_norm = d_norm.clamp(0.0, 1.0)                              # (B,)
        diff = d_norm.unsqueeze(-1) - self.centers.unsqueeze(0)       # (B, num_rbf)
        rbf = torch.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))      # (B, num_rbf)
        return torch.cat([d_norm.unsqueeze(-1), rbf], dim=-1)         # (B, 2K)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,
                 heading_range_K=4, range_max=132.0, range_min=-132.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # ---- Heading + Range conditioning (replaces CLIP text encoder) ----
        self.heading_range_K = heading_range_K
        context_dim = control_stage_config.params.get('context_dim', 768)
        self.context_dim = context_dim

        self.heading_encoder = HeadingSinusoidalEncoder(K=heading_range_K)
        self.range_encoder = RangeRBFEncoder(K=heading_range_K,
                                             range_max=range_max,
                                             range_min=range_min)

        # Projection MLP: 4K  →  77 * context_dim  (match CLIP sequence length)
        self.num_tokens = 77
        enc_dim = 4 * heading_range_K
        self.cond_projection = nn.Sequential(
            nn.Linear(enc_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, self.num_tokens * context_dim),
        )

    # --------------------------------------------------------------------- #
    #  Encode heading + range into a (B, 77, context_dim) context tensor      #
    # --------------------------------------------------------------------- #
    def encode_heading_range(self, heading_num, range_num):
        """Produce cross-attention context from scalar heading & range values.

        Args:
            heading_num: (B,) heading in degrees.
            range_num:   (B,) range / distance value.
        Returns:
            (B, 77, context_dim) tensor ready for cross-attention.
        """
        h_enc = self.heading_encoder(heading_num)    # (B, 2K)
        r_enc = self.range_encoder(range_num)        # (B, 2K)
        enc = torch.cat([h_enc, r_enc], dim=-1)      # (B, 4K)
        proj = self.cond_projection(enc)              # (B, 77 * context_dim)
        return proj.reshape(-1, self.num_tokens, self.context_dim)  # (B, 77, context_dim)

    # --------------------------------------------------------------------- #
    #  get_input  –  bypasses CLIP, encodes heading + range directly         #
    # --------------------------------------------------------------------- #
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # --- encode image to latent z (no grad – frozen first stage) ---
        with torch.no_grad():
            x = batch[self.first_stage_key]
            if bs is not None:
                x = x[:bs]
            x = x.to(self.device)
            if len(x.shape) == 3:
                x = x[..., None]
            x = einops.rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        # --- heading + range conditioning (needs grad for projection + sigma) ---
        heading_num = batch['heading_num']
        range_num = batch['range_num']
        if bs is not None:
            heading_num = heading_num[:bs]
            range_num = range_num[:bs]
        heading_num = heading_num.to(self.device).float()
        range_num = range_num.to(self.device).float()
        c = self.encode_heading_range(heading_num, range_num)  # (B, 1, context_dim)

        # --- control (hint) image ---
        with torch.no_grad():
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()

        return z, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        # Zero context as "unconditional" for classifier-free guidance
        return torch.zeros(N, self.num_tokens, self.context_dim, device=self.device)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0

        # Build human-readable text for visualisation from heading / range
        heading_vals = batch['heading_num'][:N]
        range_vals = batch['range_num'][:N]
        vis_texts = [f"h={float(heading_vals[i]):.1f}  r={float(range_vals[i]):.1f}"
                     for i in range(N)]
        log["conditioning"] = log_txt_as_img((512, 512), vis_texts, size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        # Include heading / range encoder + projection parameters
        params += list(self.heading_encoder.parameters())
        params += list(self.range_encoder.parameters())
        params += list(self.cond_projection.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.heading_encoder = self.heading_encoder.cuda()
            self.range_encoder = self.range_encoder.cuda()
            self.cond_projection = self.cond_projection.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            if self.cond_stage_model is not None:
                self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.heading_encoder = self.heading_encoder.cpu()
            self.range_encoder = self.range_encoder.cpu()
            self.cond_projection = self.cond_projection.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            if self.cond_stage_model is not None:
                self.cond_stage_model = self.cond_stage_model.cuda()
