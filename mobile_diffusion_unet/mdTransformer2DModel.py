from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import is_torch_version, logging

from mdBasicTransformerBlock import BasicTransformerMobileBlock

@dataclass
class Transformer2DMobileModelOutput(BaseOutput):
    sample: torch.FloatTensor

class Transformer2DMobileModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        # sample_size: Optional[int] = None,
        # num_vector_embeds: Optional[int] = None,
        # patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        # num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        # double_self_attention: bool = False,
        upcast_attention: bool = False,
        # norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        # attention_type: str = "default",
        # caption_channels: int = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        # self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        # self.is_input_vectorized = num_vector_embeds is not None
        # self.is_input_patches = in_channels is not None and patch_size is not None

        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = linear_cls(in_channels, inner_dim)
        else:
            self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        
        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerMobileBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    # num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    # double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    # norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    # attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_linear_projection:
            self.proj_out = linear_cls(inner_dim, in_channels)
        else:
            self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.gradient_checkpointing = False
        


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # 1. Input
        if not self.use_linear_projection:
            hidden_states = (
                self.proj_in(hidden_states, scale=lora_scale)
                if not USE_PEFT_BACKEND
                else self.proj_in(hidden_states)
            )
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = (
                self.proj_in(hidden_states, scale=lora_scale)
                if not USE_PEFT_BACKEND
                else self.proj_in(hidden_states)
            )
        
        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
        
        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = (
                self.proj_out(hidden_states, scale=lora_scale)
                if not USE_PEFT_BACKEND
                else self.proj_out(hidden_states)
            )
        else:
            hidden_states = (
                self.proj_out(hidden_states, scale=lora_scale)
                if not USE_PEFT_BACKEND
                else self.proj_out(hidden_states)
            )
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)
        
        return Transformer2DMobileModelOutput(sample=output)
