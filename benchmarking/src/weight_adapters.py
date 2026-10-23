"""Load pretrained backbones (HF hub, SensorMAE, MultiMAE) into the multimodal ViT."""

import re

import torch
from transformers import ViTModel

# timm block -> HF ViTLayer parameter names (q, k, v come from the fused qkv)
_BLOCK_KEYS = [
    ("attn.proj.weight", "attention.output.dense.weight"),
    ("attn.proj.bias", "attention.output.dense.bias"),
    ("norm1.weight", "layernorm_before.weight"),
    ("norm1.bias", "layernorm_before.bias"),
    ("norm2.weight", "layernorm_after.weight"),
    ("norm2.bias", "layernorm_after.bias"),
    ("mlp.fc1.weight", "intermediate.dense.weight"),
    ("mlp.fc1.bias", "intermediate.dense.bias"),
    ("mlp.fc2.weight", "output.dense.weight"),
    ("mlp.fc2.bias", "output.dense.bias"),
]


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint.get("state_dict", checkpoint))


def _copy_timm_blocks(model, state_dict):
    """Copy timm-style `blocks.<i>.*` weights into the HF ViT encoder, then the final norm."""
    target = model.backbone.state_dict()
    for i in range(len(model.backbone.encoder.layer)):
        src_p, tgt_p = f"blocks.{i}", f"encoder.layer.{i}"
        if f"{src_p}.norm1.weight" not in state_dict:
            continue

        q_w, k_w, v_w = state_dict[f"{src_p}.attn.qkv.weight"].chunk(3, 0)
        q_b, k_b, v_b = state_dict[f"{src_p}.attn.qkv.bias"].chunk(3, 0)
        for name, w, b in (("query", q_w, q_b), ("key", k_w, k_b), ("value", v_w, v_b)):
            target[f"{tgt_p}.attention.attention.{name}.weight"].copy_(w)
            target[f"{tgt_p}.attention.attention.{name}.bias"].copy_(b)
        for src_key, tgt_key in _BLOCK_KEYS:
            target[f"{tgt_p}.{tgt_key}"].copy_(state_dict[f"{src_p}.{src_key}"])

    if "norm.weight" in state_dict:
        model.backbone.layernorm.weight.data = state_dict["norm.weight"]
        model.backbone.layernorm.bias.data = state_dict["norm.bias"]
    elif "fc_norm.weight" in state_dict:
        model.backbone.layernorm.weight.data = state_dict["fc_norm.weight"]
        model.backbone.layernorm.bias.data = state_dict["fc_norm.bias"]


def load_standard_weights(model, backbone_name, modalities, modality_channels):
    """HF hub ViT: RGB uses the patch projection, other modalities its channel mean."""
    src_model = ViTModel.from_pretrained(backbone_name)
    model.backbone.load_state_dict(src_model.state_dict(), strict=False)

    src_proj = src_model.embeddings.patch_embeddings.projection
    adapter_weight = src_proj.weight.data
    adapter_bias = src_proj.bias.data

    avg_w = adapter_weight.mean(dim=1, keepdim=True)
    for modality in modalities:
        if modality == "rgb":
            model.rgb_adapter.weight.data = adapter_weight.clone()
            model.rgb_adapter.bias.data = adapter_bias.clone()
        else:
            adapter = getattr(model, f"{modality}_adapter")
            adapter.weight.data = avg_w.expand(-1, modality_channels[modality], -1, -1).clone()
            adapter.bias.data = adapter_bias.clone()


def _convert_sensormae_format(state_dict, hidden_dim, patch_size):
    """Map the `encoders.* / fusion_block.*` export layout onto the
    `patch_embed.* / pos_embed.* / blocks.*` layout used below."""
    if not any(k.startswith("encoders.") for k in state_dict):
        return state_dict

    converted = dict(state_dict)

    modality_names = {k.split(".")[1] for k in state_dict if k.startswith("encoders.")}
    for name in modality_names:
        proj_w = state_dict.get(f"encoders.{name}.proj.weight")
        proj_b = state_dict.get(f"encoders.{name}.proj.bias")
        if proj_w is not None and proj_b is not None:
            in_features = proj_w.shape[1]
            channels = in_features // (patch_size * patch_size)
            if channels * patch_size * patch_size == in_features:
                converted[f"patch_embed.{name}.proj.weight"] = proj_w.reshape(
                    proj_w.shape[0], channels, patch_size, patch_size
                )
                converted[f"patch_embed.{name}.proj.bias"] = proj_b

        mod_emb = state_dict.get(f"encoders.{name}.mod_emb")
        if mod_emb is not None:
            converted[f"modality_embed.{name}"] = mod_emb

        pos_emb = state_dict.get(f"encoders.{name}.pos_emb")
        if pos_emb is not None:
            converted[f"pos_embed.{name}"] = (
                pos_emb.flatten(2).transpose(1, 2) if pos_emb.dim() == 4 else pos_emb
            )

    layer_ids = {
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"fusion_block\.encoder\.layers\.(\d+)\.", k))
    }
    key_pairs = [
        ("self_attn.in_proj_weight", "attn.qkv.weight"),
        ("self_attn.in_proj_bias", "attn.qkv.bias"),
        ("self_attn.out_proj.weight", "attn.proj.weight"),
        ("self_attn.out_proj.bias", "attn.proj.bias"),
        ("linear1.weight", "mlp.fc1.weight"),
        ("linear1.bias", "mlp.fc1.bias"),
        ("linear2.weight", "mlp.fc2.weight"),
        ("linear2.bias", "mlp.fc2.bias"),
        ("norm1.weight", "norm1.weight"),
        ("norm1.bias", "norm1.bias"),
        ("norm2.weight", "norm2.weight"),
        ("norm2.bias", "norm2.bias"),
    ]
    for i in layer_ids:
        src, tgt = f"fusion_block.encoder.layers.{i}", f"blocks.{i}"
        for src_suffix, tgt_suffix in key_pairs:
            src_key = f"{src}.{src_suffix}"
            if src_key in state_dict:
                converted[f"{tgt}.{tgt_suffix}"] = state_dict[src_key]

    if "fusion_block.norm.weight" in state_dict:
        converted["norm.weight"] = state_dict["fusion_block.norm.weight"]
        converted["norm.bias"] = state_dict["fusion_block.norm.bias"]

    if "global_tokens" in state_dict:
        converted["cls_token"] = state_dict["global_tokens"]

    return converted


def _sensormae_sources(modality):
    """Checkpoint modality names that can initialize `modality` (radar/lidar use depth)."""
    candidates = [modality]
    if modality == "lwir":
        candidates.extend(["lmwir", "thermal"])
    if modality in ["nir", "swir"]:
        candidates.append("snir")
    return ["depth" if c in ("radar", "lidar") else c for c in candidates]


def load_sensormae_weights(model, checkpoint_path, modalities):
    state_dict = _load_checkpoint(checkpoint_path)
    state_dict = _convert_sensormae_format(state_dict, model.hidden_dim, model.patch_size)

    for modality in modalities:
        sources = _sensormae_sources(modality)

        for source in sources:
            weight_key = f"patch_embed.{source}.proj.weight"
            if weight_key in state_dict:
                adapter = getattr(model, f"{modality}_adapter")
                adapter.weight.data = state_dict[weight_key]
                adapter.bias.data = state_dict[f"patch_embed.{source}.proj.bias"]
                break
        else:
            print(f"Warning: no adapter for '{modality}' in {checkpoint_path}")

        for source in sources:
            mod_key = f"modality_embed.{source}"
            if mod_key in state_dict:
                getattr(model, f"modality_embed_{modality}").data = state_dict[mod_key]
                break
        else:
            print(f"Warning: no modality embedding for '{modality}' in {checkpoint_path}")

        if modality == "rgb":
            pos_rgb_key = "pos_embed.rgb"
            if pos_rgb_key not in state_dict and "pos_embed" in state_dict:
                pos_rgb_key = "pos_embed"

            if pos_rgb_key in state_dict:
                src_pos_rgb = state_dict[pos_rgb_key]
                if hasattr(model, "pos_embed_rgb") and src_pos_rgb.shape == model.pos_embed_rgb.shape:
                    model.pos_embed_rgb.data = src_pos_rgb.clone()
                elif (
                    hasattr(model.backbone, "embeddings")
                    and src_pos_rgb.shape == model.backbone.embeddings.position_embeddings.shape
                ):
                    model.backbone.embeddings.position_embeddings.data = src_pos_rgb.clone()
        else:
            loaded_pos = False
            for source in sources:
                pos_key = f"pos_embed.{source}"
                if pos_key in state_dict:
                    src_pos = state_dict[pos_key]
                    pos_embed = getattr(model, f"pos_embed_{modality}", None)
                    if pos_embed is not None and src_pos.shape == pos_embed.shape:
                        pos_embed.data = src_pos.clone()
                        loaded_pos = True
                    break

            if not loaded_pos and "pos_embed" in state_dict:
                src_pos = state_dict["pos_embed"]
                pos_embed = getattr(model, f"pos_embed_{modality}", None)
                if pos_embed is not None and src_pos.shape == pos_embed.shape:
                    pos_embed.data = src_pos.clone()

    model.backbone.load_state_dict(state_dict, strict=False)
    _copy_timm_blocks(model, state_dict)

    if "cls_token" in state_dict:
        model.backbone.embeddings.cls_token.data = state_dict["cls_token"]


# MultiMAE input adapters that are plain strided convolutions like ours
NATIVE_COMPATIBLE_MODALITIES = {"rgb", "depth"}


def load_multimae_weights(model, checkpoint_path, modalities, modality_channels):
    """MultiMAE checkpoint, either the "multivit" export (per-modality input adapters;
    rgb / depth are loaded natively) or the "timm" export (RGB only). Other modalities
    are initialized with the channel mean of the RGB patch projection."""
    raw_state_dict = _load_checkpoint(checkpoint_path)

    is_multivit = any(k.startswith("input_adapters.") for k in raw_state_dict)

    if is_multivit:
        state_dict = {}
        for k, v in raw_state_dict.items():
            k = k.replace("module.", "")
            m = re.match(r"^encoder\.(\d+)\.(.*)$", k)
            if m:
                k = f"blocks.{m.group(1)}.{m.group(2)}"
            state_dict[k] = v
    else:
        state_dict = {
            k.replace("module.", "").replace("backbone.", "").replace("encoder.", ""): v
            for k, v in raw_state_dict.items()
        }

    if is_multivit:
        rgb_w = state_dict.get("input_adapters.rgb.proj.weight")
        rgb_b = state_dict.get("input_adapters.rgb.proj.bias")
        for modality in modalities:
            adapter = getattr(model, f"{modality}_adapter")
            native_w = state_dict.get(f"input_adapters.{modality}.proj.weight")
            native_b = state_dict.get(f"input_adapters.{modality}.proj.bias")
            if modality in NATIVE_COMPATIBLE_MODALITIES and native_w is not None:
                adapter.weight.data = native_w.clone()
                adapter.bias.data = native_b.clone()
            elif modality == "rgb":
                adapter.weight.data = rgb_w.clone()
                adapter.bias.data = rgb_b.clone()
            else:
                avg_w = rgb_w.mean(dim=1, keepdim=True)
                adapter.weight.data = avg_w.expand(-1, modality_channels[modality], -1, -1).clone()
                adapter.bias.data = rgb_b.clone()
    elif "patch_embed.proj.weight" in state_dict:
        avg_w = state_dict["patch_embed.proj.weight"].mean(dim=1, keepdim=True)
        bias = state_dict["patch_embed.proj.bias"]
        for modality in modalities:
            adapter = getattr(model, f"{modality}_adapter")
            if modality == "rgb":
                adapter.weight.data = state_dict["patch_embed.proj.weight"]
                adapter.bias.data = bias.clone()
            else:
                adapter.weight.data = avg_w.expand(-1, modality_channels[modality], -1, -1).clone()
                adapter.bias.data = bias.clone()

    if is_multivit:
        rgb_pos = state_dict.get("input_adapters.rgb.pos_emb")
        if rgb_pos is not None and rgb_pos.dim() == 4:
            rgb_pos = rgb_pos.flatten(2).transpose(1, 2)
        for modality in modalities:
            pos_embed = getattr(model, f"pos_embed_{modality}", None)
            if pos_embed is None:
                continue
            native_pos = None
            if modality in NATIVE_COMPATIBLE_MODALITIES:
                native_pos = state_dict.get(f"input_adapters.{modality}.pos_emb")
                if native_pos is not None and native_pos.dim() == 4:
                    native_pos = native_pos.flatten(2).transpose(1, 2)
            src_pos = native_pos if native_pos is not None else rgb_pos
            if src_pos is not None and src_pos.shape == pos_embed.shape:
                pos_embed.data = src_pos.clone()
    elif "pos_embed" in state_dict:
        for modality in modalities:
            src_pos = state_dict["pos_embed"].clone()
            pos_embed = getattr(model, f"pos_embed_{modality}", None)
            if pos_embed is not None and src_pos.shape[1] == pos_embed.shape[1] + 1:
                src_pos = src_pos[:, 1:, :]  # drop the CLS position
            if pos_embed is not None and src_pos.shape == pos_embed.shape:
                pos_embed.data = src_pos

    cls_key = "cls_token" if "cls_token" in state_dict else "global_tokens"
    if cls_key in state_dict:
        model.backbone.embeddings.cls_token.data = state_dict[cls_key]

    model.backbone.load_state_dict(state_dict, strict=False)
    _copy_timm_blocks(model, state_dict)
