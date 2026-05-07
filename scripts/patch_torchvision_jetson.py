#!/usr/bin/env python3
"""Patch torchvision 0.19 for Jetson compatibility with NV torch 2.5.

Run as: python patch_torchvision_jetson.py <site-packages-path>

Applies three patches to work around incompatible C++ ops and symlinks
system TensorRT packages into the venv.
"""
import sys
import pathlib

site = pathlib.Path(sys.argv[1])

# 1. _meta_registrations.py — disable torch dispatch registration
(site / "torchvision" / "_meta_registrations.py").write_text("# disabled for Jetson TRT compat\n")

# 2. extension.py — make _assert_has_ops() a no-op
ext = site / "torchvision" / "extension.py"
txt = ext.read_text()
txt = txt.replace(
    "def _assert_has_ops():\n    if not _has_ops():",
    "def _assert_has_ops():\n    return\n    if not _has_ops():",
)
ext.write_text(txt)

# 3. ops/boxes.py — pure-torch NMS fallback when C++ ops unavailable
boxes = site / "torchvision" / "ops" / "boxes.py"
txt = boxes.read_text()
old = "    _assert_has_ops()\n    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)"
new = '''    from torchvision.extension import _has_ops
    if _has_ops():
        return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    # Pure-torch NMS fallback for Jetson (incompatible C++ ops)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter)
        mask = iou <= iou_threshold
        order = rest[mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)'''
txt = txt.replace(old, new)
boxes.write_text(txt)

# 4. Symlink system TensorRT into venv
trt_src = pathlib.Path("/usr/lib/python3.10/dist-packages")
for name in ("tensorrt", "tensorrt_lean", "tensorrt_dispatch"):
    for suffix in ("", "-10.3.0.dist-info"):
        src = trt_src / f"{name}{suffix}"
        dst = site / f"{name}{suffix}"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

print("Jetson GPU patches applied")
