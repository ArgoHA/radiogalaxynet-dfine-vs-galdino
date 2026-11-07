"""Shared helpers for the Gal-DINO reproduction / export / TRT scripts (not part of upstream).

- build_model(): Gal-DINO built exactly as main.py does (config/DINO/DINO_4scale.py + argparse
  defaults), weights from the figshare checkpoint, eval mode.
- use_pytorch_msda(): swap the custom CUDA MultiScaleDeformableAttention op for the repo's own
  pure-PyTorch grid_sample path (ms_deform_attn_core_pytorch) - the ONNX/TRT path.
- preprocessing variants of Gal-DINO's test transform (RandomResize([800], max_size=1333) +
  ToTensor + Normalize(radio mean/std)).
- ExportWrapper: normalized image -> model -> PostProcess top-k in-graph.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MEAN = [0.06935824, 0.3000091, 0.29598746]
STD = [0.18748689, 0.4532347, 0.4564638]
MIN_SIZE, MAX_SIZE = int(os.environ.get("GALDINO_SIZE", 800)), 1333  # test-time short side
NUM_SELECT = 300
# HERE is the Gal-DINO clone (galdino/Gal-DINO); the dataset sits at <repo>/data/RadioGalaxyNET
DATA = Path(os.environ.get("RGNET_DATA", HERE.parents[1] / "data" / "RadioGalaxyNET"))


def get_args():
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        import main as galmain
        from util.slconfig import SLConfig

        parser = argparse.ArgumentParser(parents=[galmain.get_args_parser()])
        args = parser.parse_args(["-c", "config/DINO/DINO_4scale.py", "--eval"])
        cfg = SLConfig.fromfile(args.config_file)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        args.distributed = False
    finally:
        os.chdir(cwd)
    return args


def build_model(ckpt=HERE / "outputs_gal/checkpoint.pth", device="cuda"):
    from models.registry import MODULE_BUILD_FUNCS

    args = get_args()
    model, _, postprocessors = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    torch.serialization.add_safe_globals([argparse.Namespace])
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd)
    return model.to(device).eval(), postprocessors


class _PytorchMSDA:
    """Drop-in for MSDeformAttnFunction using the repo's pure-PyTorch grid_sample core."""

    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attn, im2col_step):
        shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        return ms_deform_attn_core(value, shapes, sampling_locations, attn)


def ms_deform_attn_core(value, shapes, sampling_locations, attention_weights):
    """ms_deform_attn_core_pytorch from models/dino/ops/functions/ms_deform_attn_func.py, with
    python-int level shapes so the ONNX trace has static split sizes. Same math."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    out = []
    for lid_, (H_, W_) in enumerate(shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        out.append(F.grid_sample(value_l_, grid_l_, mode="bilinear", padding_mode="zeros",
                                 align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(out, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(N_, M_ * D_, Lq_).transpose(1, 2).contiguous()


def use_pytorch_msda():
    import models.dino.deformable_transformer as dt
    import models.dino.ops.modules.ms_deform_attn as m

    m.MSDeformAttnFunction = _PytorchMSDA

    # gen_encoder_output_proposals slices the padding mask with 0-d CUDA tensors taken from
    # spatial_shapes; the TorchScript ONNX exporter segfaults on that. Hand it python ints
    # (same values, same math; shapes are static for a fixed input size).
    orig = dt.gen_encoder_output_proposals
    if getattr(orig, "_int_shapes", False):
        return

    def gen_props(memory, mask, spatial_shapes, learnedwh=None):
        shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        return orig(memory, mask, shapes, learnedwh)

    gen_props._int_shapes = True
    dt.gen_encoder_output_proposals = gen_props


def out_size(h, w, size=MIN_SIZE, max_size=MAX_SIZE):
    """datasets/transforms.py get_size_with_aspect_ratio."""
    mn, mx = float(min(w, h)), float(max(w, h))
    if mx / mn * size > max_size:
        size = int(round(max_size * mn / mx))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def preprocess_pil(path):
    """Gal-DINO's exact eval transform (their own code): PIL RGB -> T.RandomResize([800], 1333)
    -> ToTensor -> Normalize. Returns [3, H, W] float32 CPU."""
    from PIL import Image
    import datasets.transforms as T

    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.RandomResize([MIN_SIZE], max_size=MAX_SIZE), T.ToTensor(),
                    T.Normalize(MEAN, STD)])
    return tf(img, None)[0]


class GpuPreprocess:
    """uint8 HxWx3 BGR (cv2.imread) -> pinned -> H2D -> bilinear resize + RGB + normalize on GPU,
    written into `dst` ([1, 3, h, w] fp32 engine input). Resize is half-pixel bilinear without
    antialias, which is what PIL bilinear does for this upscale (450 -> 800)."""

    def __init__(self, h, w, dst, device="cuda"):
        self.pinned = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
        self.gpu = torch.empty((h, w, 3), dtype=torch.uint8, device=device)
        self.dst = dst
        oh, ow = out_size(h, w)
        self.size = (oh, ow)
        assert tuple(dst.shape[-2:]) == (oh, ow), (dst.shape, oh, ow)
        # BGR channel order folded into the normalization constants
        self.scale = (1.0 / (255.0 * torch.tensor(STD[::-1], device=device))).view(1, 3, 1, 1)
        self.shift = (-torch.tensor(MEAN[::-1], device=device) / torch.tensor(
            STD[::-1], device=device)).view(1, 3, 1, 1)

    def __call__(self, img_bgr):
        np.copyto(self.pinned.numpy(), img_bgr)
        self.gpu.copy_(self.pinned, non_blocking=True)
        x = self.gpu.permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = torch.addcmul(self.shift, x, self.scale)  # normalized, still BGR
        self.dst.copy_(x.flip(1))


def preprocess_cv2(img_bgr):
    """CPU alternative: cv2 INTER_LINEAR resize + normalize. [3, H, W] float32."""
    import cv2

    h, w = img_bgr.shape[:2]
    oh, ow = out_size(h, w)
    x = cv2.resize(img_bgr, (ow, oh), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    x = (x.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ExportWrapper(torch.nn.Module):
    """[1, 3, H, W] normalized -> Gal-DINO -> PostProcess / PostProcesskeypoint top-k, in graph.

    Outputs (normalized to [0, 1], multiply by the original W/H on the client):
      scores [1, K] fp32, labels [1, K] int32, boxes [1, K, 4] xyxy, kpts [1, K, 2] xy.
    Same ops as models/dino/dino.py PostProcess: sigmoid, top-k over queries*classes, query =
    idx // C, label = idx % C, gather, cxcywh -> xyxy.
    """

    def __init__(self, model, num_select=NUM_SELECT):
        super().__init__()
        self.model = model
        self.k = num_select

    def forward(self, x):
        from util.misc import NestedTensor

        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool,
                           device=x.device)
        out = self.model(NestedTensor(x, mask))
        logits, boxes, kpts = out["pred_logits"], out["pred_boxes"], out["pred_keypoints"]
        logits, boxes, kpts = logits.float(), boxes.float(), kpts.float()
        C = logits.shape[2]
        scores, idx = torch.topk(logits.sigmoid().flatten(1), self.k, dim=1)
        q = torch.div(idx, C, rounding_mode="floor")
        labels = (idx - q * C).to(torch.int32)
        boxes = torch.gather(boxes, 1, q.unsqueeze(-1).repeat(1, 1, 4))
        cx, cy, w, h = boxes.unbind(-1)
        boxes = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
        kpts = torch.gather(kpts, 1, q.unsqueeze(-1).repeat(1, 1, 2))
        return scores, labels, boxes, kpts


def to_coco(image_id, scores, labels, boxes, kpts, w, h):
    """Normalized top-k outputs (numpy, one image) -> COCO results (Gal-DINO's prepare_*)."""
    s = np.array([w, h, w, h], np.float32)
    b = boxes * s
    k = kpts * np.array([w, h], np.float32)
    res = []
    for i in range(len(scores)):
        x0, y0, x1, y1 = b[i].tolist()
        res.append({"image_id": int(image_id), "category_id": int(labels[i]),
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(scores[i]),
                    "keypoints": [float(k[i, 0]), float(k[i, 1]), 2]})
    return res
