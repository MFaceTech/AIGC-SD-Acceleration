from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.activations import get_activation
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from mdDownBlock2D import DownBlock2DMobile
from mdCADownBlock import CADownBlock2DMobile
from mdCAUpBlock import CAUpBlock2DMobile
from mdUpBlock2D import UpBlock2DMobile

@dataclass
class UNet2DMobileModelOutput(BaseOutput):
    sample: torch.FloatTensor = None

class UNet2DMobileModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        # center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        # down_block_types: Tuple[str] = (
        #     "CrossAttnDownBlock2D",
        #     "CrossAttnDownBlock2D",
        #     "CrossAttnDownBlock2D",
        #     "DownBlock2D",
        # ),
        # mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        # up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        # only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1024),
        # layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        # mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        # transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        # reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        # encoder_hid_dim: Optional[int] = None,
        # encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        # dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        # class_embed_type: Optional[str] = None,
        # addition_embed_type: Optional[str] = None,
        # addition_time_embed_dim: Optional[int] = None,
        # num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        # resnet_skip_time_act: bool = False,
        # resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        # projection_class_embeddings_input_dim: Optional[int] = None,
        # attention_type: str = "default",
        # class_embeddings_concat: bool = False,
        # mid_block_only_cross_attention: Optional[bool] = None,
        # cross_attention_norm: Optional[str] = None,
        # addition_embed_type_num_heads=64,
    ):
        super().__init__()

        self.sample_size = sample_size
        num_attention_heads = num_attention_heads or attention_head_dim

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self.encoder_hid_proj = None

        # class embedding
        self.class_embedding = None

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        # only_cross_attention = False
        # mid_block_only_cross_attention = False
        # only_cross_attention = [only_cross_attention] * len(down_block_types)

        # if isinstance(num_attention_heads, int):    # todo: 修改down_block_types
        #     num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # if isinstance(attention_head_dim, int):
        #     attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # if isinstance(cross_attention_dim, int):
        #     cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        # if isinstance(layers_per_block, int):   # todo: [1, 1, 5, 2, 2]
        #     layers_per_block = [layers_per_block] * len(down_block_types)
        
        # if isinstance(transformer_layers_per_block, int):   # todo: [0, CA, CA+SA]
        #     transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        
        blocks_time_embed_dim = time_embed_dim

        # down1
        self.downblock1 = DownBlock2DMobile(
            num_layers = 1,
            in_channels = block_out_channels[0],
            out_channels = block_out_channels[0],
            temb_channels = blocks_time_embed_dim,
            dropout = dropout,
            add_downsample = True,
            resnet_eps = norm_eps,
            resnet_act_fn = act_fn,
            resnet_groups = norm_num_groups,
            downsample_padding = downsample_padding,
            resnet_time_scale_shift = resnet_time_scale_shift,
        )
        # down2
        self.downblock2 = CADownBlock2DMobile(
            in_channels = block_out_channels[0],
            out_channels = block_out_channels[1],
            temb_channels = blocks_time_embed_dim,
            dropout = dropout,
            num_layers = 1,
            transformer_layers_per_block = 1,
            resnet_eps = norm_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_act_fn = act_fn,
            resnet_groups = norm_num_groups,
            # resnet_pre_norm,
            num_attention_heads = num_attention_heads,
            cross_attention_dim = cross_attention_dim,
            # output_scale_factor,
            downsample_padding = downsample_padding,
            add_downsample = True,
            use_linear_projection = use_linear_projection,
            only_cross_attention = True,    # 只有CA
            upcast_attention = upcast_attention,
        )

        # mid
        self.midblock = CADownBlock2DMobile(
            in_channels = block_out_channels[1],
            out_channels = block_out_channels[2],
            temb_channels = blocks_time_embed_dim,
            dropout = dropout,
            num_layers = 5,
            transformer_layers_per_block = 3,   # 每层3个att
            resnet_eps = norm_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_act_fn = act_fn,
            resnet_groups = norm_num_groups,
            # resnet_pre_norm,
            num_attention_heads = num_attention_heads,
            cross_attention_dim = cross_attention_dim,
            # output_scale_factor,
            downsample_padding = downsample_padding,
            add_downsample = False,     # 不需下采样
            use_linear_projection = use_linear_projection,
            only_cross_attention = False,    # SA+CA
            upcast_attention = upcast_attention,
        )

        # up1
        self.upblock1 = CAUpBlock2DMobile(
            in_channels = block_out_channels[1] + block_out_channels[2],
            out_channels = block_out_channels[1],
            # prev_output_channel = block_out_channels[2],
            temb_channels = blocks_time_embed_dim,
            # resolution_idx: Optional[int] = None,
            dropout = dropout,
            num_layers = 2,
            transformer_layers_per_block = 2,   # 每层2个att的CA
            resnet_eps = norm_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_act_fn = act_fn,
            resnet_groups = norm_num_groups,
            # resnet_pre_norm: bool = True,
            num_attention_heads = num_attention_heads,
            cross_attention_dim = cross_attention_dim,
            # output_scale_factor: float = 1.0,
            add_upsample = True,
            # dual_cross_attention: bool = False,
            use_linear_projection = use_linear_projection,
            only_cross_attention = True,    # 只有CA
            upcast_attention = upcast_attention,
            # attention_type: str = "default",
        )

        # up2
        self.upblock2 = UpBlock2DMobile(
            in_channels = block_out_channels[0] + block_out_channels[1],
            # prev_output_channel = block_out_channels[1],
            out_channels = block_out_channels[0],
            temb_channels = blocks_time_embed_dim,
            # resolution_idx: Optional[int] = None,
            dropout = dropout,
            num_layers = 2,
            resnet_eps = norm_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_act_fn = act_fn,
            resnet_groups = norm_num_groups,
            # resnet_pre_norm: bool = True,
            # output_scale_factor: float = 1.0,
            add_upsample = True,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )

        self.conv_act = get_activation(act_fn)

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )
    
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DMobileModelOutput, Tuple]:
        if attention_mask is not None:
            pass
        if encoder_attention_mask is not None:
            pass

        # 0. center input if necessary
        # if self.config.center_input_sample:
        #     sample = 2 * sample - 1.0
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            pass

        # if self.config.addition_embed_type: pass

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        
        # if self.encoder_hid_proj: pass

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        
        # is_controlnet
        # is_adapter
        sample_down1 = self.downblock1(
            hidden_states = sample,
            temb = emb,
            scale = lora_scale
        )

        sample_down2 = self.downblock2(
            hidden_states = sample_down1,
            temb = emb,
            encoder_hidden_states = encoder_hidden_states,
            attention_mask = attention_mask,
            cross_attention_kwargs = cross_attention_kwargs,
            encoder_attention_mask = encoder_attention_mask
        )

        # if is_controlnet: pass

        sample_mid = self.midblock(
            hidden_states = sample_down2,
            temb = emb,
            encoder_hidden_states = encoder_hidden_states,
            attention_mask = attention_mask,
            cross_attention_kwargs = cross_attention_kwargs,
            encoder_attention_mask = encoder_attention_mask
        )
        
        # if is_controlnet: pass

        sample_up1 = self.upblock1(
            hidden_states = sample_mid,
            temb = emb,
            res_hidden_states = sample_down2,
            encoder_hidden_states = encoder_hidden_states,
            cross_attention_kwargs = cross_attention_kwargs,
            attention_mask = attention_mask,
            encoder_attention_mask = encoder_attention_mask,
        )

        sample = self.upblock2(
            hidden_states = sample_up1,
            res_hidden_states = sample_down1,
            temb = emb,
            scale = lora_scale,
        )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        if not return_dict:
            return (sample,)
    
        return UNet2DMobileModelOutput(sample=sample)
    