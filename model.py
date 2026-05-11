
# ============================================================
# Spatial-Frequency Gated Swin Transformer for Remote Sensing Single Image Super-Resolution
# SFG-SwinSR
# SFG-SwinSR Model Code
# Exact version: replace FFN inside Swin2SR Transformer blocks
# Lightweight updated version without changing class names
# ============================================================

import types
import torch
import torch.nn as nn

from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

# ============================================================
# Window partition / reverse
# Same logic used inside Swin/Swin2SR
# ============================================================

def window_partition(x, window_size):
    """
    x: [B, H, W, C]
    return: windows [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: [num_windows*B, window_size, window_size, C]
    return: x [B, H, W, C]
    """
    C = windows.shape[-1]

    B = int(windows.shape[0] / (H * W / window_size / window_size))

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C,
    )

    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, C)
    )

    return x


# ============================================================
# Spatial Frequency Gated FFN
# Lightweight version
# This replaces the original Swin2SR FFN/MLP
# ============================================================

class SpatialFrequencyGatedFFN(nn.Module):
    """
    Lightweight Spatial Frequency gated feed-forward network.

    Replaces:
        Swin2SRIntermediate + Swin2SROutput

    Input:
        x: [B, H*W, C]

    Output:
        out: [B, H*W, C]

    Main idea:
        1. Expand channel dimension.
        2. Estimate SFG-like low-pass response using depthwise blur.
        3. Compute high-frequency residual.
        4. Use lightweight bottleneck gate to control detail injection.
        5. Project back to original dimension.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=2.0,
        drop=0.0,
        sfg_kernel_size=5,
        gate_reduction=8,
    ):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)
        gate_dim = max(hidden_dim // gate_reduction, 16)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.sfg_kernel_size = sfg_kernel_size

        # Expansion, same role as original FFN first Linear
        self.fc1 = nn.Linear(dim, hidden_dim)

        self.content_act = nn.GELU()

        # SFG-like low-pass filter: depthwise, lightweight
        self.sfg_blur = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=sfg_kernel_size,
            padding=sfg_kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )

        # Initialize as average blur
        nn.init.constant_(
            self.sfg_blur.weight,
            1.0 / (sfg_kernel_size * sfg_kernel_size),
        )

        # Lightweight detail branch:
        # depthwise conv only, no heavy hidden_dim -> hidden_dim pointwise conv
        self.detail_branch = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=True,
            ),
            nn.GELU(),
        )

        # Lightweight bottleneck gate
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, gate_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Projection, same role as original FFN second Linear
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        x: [B, H*W, C]
        H, W: spatial size of token feature map
        """

        B, N, C = x.shape
        assert N == H * W, f"Token number {N} does not match H*W={H*W}"

        # ----------------------------------------------------
        # 1. FFN expansion
        # ----------------------------------------------------
        x = self.fc1(x)                 # [B, H*W, hidden_dim]
        x = self.content_act(x)

        # ----------------------------------------------------
        # 2. Convert token feature to spatial feature
        # ----------------------------------------------------
        feat = (
            x.transpose(1, 2)
            .contiguous()
            .view(B, self.hidden_dim, H, W)
        )                               # [B, hidden_dim, H, W]

        # ----------------------------------------------------
        # 3. SFG-like low-pass response
        # ----------------------------------------------------
        blur_feat = self.sfg_blur(feat)

        # ----------------------------------------------------
        # 4. High-frequency residual
        # ----------------------------------------------------
        detail_feat = feat - blur_feat

        # ----------------------------------------------------
        # 5. Lightweight detail refinement
        # ----------------------------------------------------
        detail_feat = self.detail_branch(detail_feat)

        # ----------------------------------------------------
        # 6. Adaptive bottleneck gate
        # ----------------------------------------------------
        gate = self.gate(detail_feat)

        # ----------------------------------------------------
        # 7. Gated feature fusion
        # ----------------------------------------------------
        fused_feat = feat + gate * detail_feat

        # ----------------------------------------------------
        # 8. Convert back to token format
        # ----------------------------------------------------
        fused_tokens = (
            fused_feat.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )                               # [B, H*W, hidden_dim]

        # ----------------------------------------------------
        # 9. Project back to original channel dimension
        # ----------------------------------------------------
        out = self.fc2(fused_tokens)
        out = self.drop(out)

        return out


# ============================================================
# New forward for Swin2SRLayer
# This keeps attention same, replaces only FFN part
# ============================================================

def mag_swin2sr_layer_forward(
    self,
    hidden_states,
    input_dimensions,
    head_mask=None,
    output_attentions=False,
):
    """
    Modified Swin2SRLayer forward.

    Original:
        attention -> standard FFN

    Proposed:
        attention -> Spatial Frequency gated FFN

    hidden_states: [B, H*W, C]
    input_dimensions: (H, W)
    """

    height, width = input_dimensions
    batch_size, _, channels = hidden_states.size()

    shortcut = hidden_states

    # --------------------------------------------------------
    # 1. Reshape tokens to spatial feature
    # --------------------------------------------------------
    hidden_states = hidden_states.view(batch_size, height, width, channels)

    # --------------------------------------------------------
    # 2. Pad to multiples of window size
    # --------------------------------------------------------
    hidden_states, pad_values = self.maybe_pad(
        hidden_states,
        height,
        width,
    )

    _, height_pad, width_pad, _ = hidden_states.shape

    # --------------------------------------------------------
    # 3. Shifted window operation
    # --------------------------------------------------------
    if self.shift_size > 0:
        shifted_hidden_states = torch.roll(
            hidden_states,
            shifts=(-self.shift_size, -self.shift_size),
            dims=(1, 2),
        )
    else:
        shifted_hidden_states = hidden_states

    # --------------------------------------------------------
    # 4. Window partition
    # --------------------------------------------------------
    hidden_states_windows = window_partition(
        shifted_hidden_states,
        self.window_size,
    )

    hidden_states_windows = hidden_states_windows.view(
        -1,
        self.window_size * self.window_size,
        channels,
    )

    # --------------------------------------------------------
    # 5. Attention mask
    # --------------------------------------------------------
    attn_mask = self.get_attn_mask(
        height_pad,
        width_pad,
        dtype=hidden_states.dtype,
    )

    if attn_mask is not None:
        attn_mask = attn_mask.to(hidden_states_windows.device)

    # --------------------------------------------------------
    # 6. Original Swin2SR window attention
    # --------------------------------------------------------
    attention_outputs = self.attention(
        hidden_states_windows,
        attn_mask,
        head_mask,
        output_attentions=output_attentions,
    )

    attention_output = attention_outputs[0]

    # --------------------------------------------------------
    # 7. Merge windows
    # --------------------------------------------------------
    attention_windows = attention_output.view(
        -1,
        self.window_size,
        self.window_size,
        channels,
    )

    shifted_windows = window_reverse(
        attention_windows,
        self.window_size,
        height_pad,
        width_pad,
    )

    # --------------------------------------------------------
    # 8. Reverse cyclic shift
    # --------------------------------------------------------
    if self.shift_size > 0:
        attention_windows = torch.roll(
            shifted_windows,
            shifts=(self.shift_size, self.shift_size),
            dims=(1, 2),
        )
    else:
        attention_windows = shifted_windows

    # --------------------------------------------------------
    # 9. Remove padding
    # --------------------------------------------------------
    was_padded = pad_values[3] > 0 or pad_values[5] > 0

    if was_padded:
        attention_windows = attention_windows[:, :height, :width, :].contiguous()

    attention_windows = attention_windows.view(
        batch_size,
        height * width,
        channels,
    )

    # --------------------------------------------------------
    # 10. Same attention residual path as Swin2SR
    # --------------------------------------------------------
    hidden_states = self.layernorm_before(attention_windows)
    hidden_states = shortcut + self.drop_path(hidden_states)

    # --------------------------------------------------------
    # 11. Proposed MAG-FFN instead of original FFN
    # --------------------------------------------------------
    ffn_output = self.mag_ffn(
        hidden_states,
        height,
        width,
    )

    # --------------------------------------------------------
    # 12. Same FFN residual style as Swin2SR
    # --------------------------------------------------------
    layer_output = hidden_states + self.drop_path(
        self.layernorm_after(ffn_output)
    )

    if output_attentions:
        layer_outputs = (layer_output, attention_outputs[1])
    else:
        layer_outputs = (layer_output,)

    return layer_outputs


# ============================================================
# SFG-Swin2SR Full Model
# ============================================================

class MAGSwin2SR(nn.Module):
    """
    Exact MAG-Swin2SR.

    This model uses HuggingFace Swin2SR but replaces the internal
    FFN/MLP of every Swin2SR Transformer layer with SpatialFrequencyGatedFFN.

    Input:
        pixel_values: [B, 3, H, W]

    Output:
        sr: [B, 3, scale*H, scale*W]
    """

    def __init__(
        self,
        upscale=2,
        img_size=64,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_channels=3,
        mlp_ratio=2.0,
        sfg_kernel_size=5,
        drop=0.0,
        gate_reduction=8,
    ):
        super().__init__()

        config = Swin2SRConfig(
            num_channels=num_channels,
            upscale=upscale,
            img_size=img_size,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
        )

        self.swin2sr = Swin2SRForImageSuperResolution(config)

        # Replace FFN inside every Swin2SRLayer
        self._replace_swin2sr_ffn_with_mag_ffn(
            mlp_ratio=mlp_ratio,
            sfg_kernel_size=sfg_kernel_size,
            drop=drop,
            gate_reduction=gate_reduction,
        )

    def _replace_swin2sr_ffn_with_mag_ffn(
        self,
        mlp_ratio=2.0,
        sfg_kernel_size=5,
        drop=0.0,
        gate_reduction=8,
    ):
        """
        Find every Swin2SRLayer and replace original FFN with MAG-FFN.

        We do not change attention.
        We do not add a stem.
        We only replace the FFN path inside each Swin2SR block.

        Important:
            module.intermediate and module.output are replaced by nn.Identity()
            so the original FFN parameters are not counted.
        """

        replaced_count = 0

        for module in self.swin2sr.modules():
            if module.__class__.__name__ == "Swin2SRLayer":

                # Infer hidden dimension from original FFN before removing it
                dim = module.intermediate.dense.in_features

                # Attach proposed lightweight MAG-FFN
                module.mag_ffn = SpatialFrequencyGatedFFN(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    sfg_kernel_size=sfg_kernel_size,
                    gate_reduction=gate_reduction,
                )

                # Very important:
                # remove original FFN modules from parameter count
                module.intermediate = nn.Identity()
                module.output = nn.Identity()

                # Replace layer forward
                module.forward = types.MethodType(
                    mag_swin2sr_layer_forward,
                    module,
                )

                replaced_count += 1

        print(f"[MAG-Swin2SR] Replaced FFN in {replaced_count} Swin2SR layers.")

        if replaced_count == 0:
            raise RuntimeError(
                "No Swin2SRLayer found. Please check your transformers version."
            )

    def forward(self, pixel_values):
        """
        Forward pass.

        pixel_values:
            LR RGB satellite image [B, 3, H, W]

        returns:
            SR RGB image [B, 3, upscale*H, upscale*W]
        """

        outputs = self.swin2sr(pixel_values=pixel_values)
        return outputs.reconstruction


# ============================================================
# Forward pass test
# ============================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAGSwin2SR(
        upscale=2,
        img_size=64,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_channels=4,
        mlp_ratio=2.0,
        sfg_kernel_size=5,
        drop=0.0,
        gate_reduction=8,
    ).to(device)

    lr = torch.rand(24, 4, 64, 64).to(device)

    sr = model(pixel_values=lr)

    print("Input LR shape :", lr.shape)
    print("Output SR shape:", sr.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters    : {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
