#!/usr/bin/env python3
"""Memory profiler for DocRes inference on MPS (Apple Silicon)."""

import gc
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "mbd"))

from models.restormer_arch import Restormer
from model.deep_lab_model.deeplab import DeepLab
from safetensors.torch import load_file

DEVICE = torch.device("mps")
NUM_RUNS = 10

def mb(b):
    return f"{b / 1024 / 1024:.1f} MB"

def mem_stats():
    alloc = torch.mps.current_allocated_memory()
    drv = torch.mps.driver_allocated_memory()
    return alloc, drv

def print_mem(label):
    alloc, drv = mem_stats()
    print(f"  [{label}] allocated={mb(alloc)}  driver={mb(drv)}")

def cleanup():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

def stride_integral(img, stride=8):
    h, w = img.shape[:2]
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, pad_h, 0, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
    return img, pad_h, pad_w

def appearance_prompt(img):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (1024, 1024))
    planes = []
    for plane in cv2.split(resized):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        planes.append(norm)
    return cv2.resize(cv2.merge(planes), (w, h))

def deshadow_prompt(img):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (1024, 1024))
    planes = []
    for plane in cv2.split(resized):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        planes.append(bg)
    return cv2.resize(cv2.merge(planes), (w, h))

def deblur_prompt(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    hf = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)

def make_test_image(w=1024, h=1024):
    img = np.random.randint(50, 230, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (w-100, h-100), (255, 255, 255), -1)
    cv2.putText(img, "Test Document", (200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


print("=" * 60)
print("DocRes MPS Memory Profiler")
print("=" * 60)

print("\nLoading models...")
print_mem("before load")

model = Restormer(
    inp_channels=6, out_channels=3, dim=48,
    num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
    bias=False, LayerNorm_type="WithBias", dual_pixel_task=True,
)
model.load_state_dict(load_file("checkpoints/docres.safetensors"))
model.eval()
model = model.half().to(DEVICE)
print_mem("restormer loaded (half)")

mbd_model = DeepLab(num_classes=1, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
mbd_model.load_state_dict(load_file("checkpoints/mbd.safetensors"))
mbd_model.eval()
mbd_model = mbd_model.half().to(DEVICE)
print_mem("mbd loaded (half)")

cleanup()
print_mem("after cleanup")


def run_model(img_bgr, prompt_fn, max_dim=1024):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    prompt = prompt_fn(img_bgr)
    in_im = np.concatenate((img_bgr, prompt), -1)
    in_im, pad_h, pad_w = stride_integral(in_im, 8)
    in_im = torch.from_numpy((in_im / 255.0).transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    try:
        with torch.no_grad():
            pred = model(in_im)
            pred = torch.clamp(pred, 0, 1)
            pred = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    finally:
        del in_im
        cleanup()
    return pred[pad_h:, pad_w:]

def get_mask(img_bgr):
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (448, 448))
    img = cv2.GaussianBlur(img, (15, 15), 0, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy((img.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    try:
        with torch.no_grad():
            pred = mbd_model(img_t)
            mask = pred[:, 0, :, :].unsqueeze(1)
            mask = F.interpolate(mask, (h, w))
            mask = mask.squeeze(0).squeeze(0).cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
    finally:
        del img_t
        cleanup()
    kernel = np.ones((3, 3))
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask[mask > 100] = 255
    mask[mask < 100] = 0
    return mask


print("\n" + "=" * 60)
print(f"ENHANCE pipeline x{NUM_RUNS} (deshadow + appearance + sharpen)")
print("=" * 60)

img = make_test_image()
baseline_alloc, baseline_drv = mem_stats()
print(f"Baseline: allocated={mb(baseline_alloc)}  driver={mb(baseline_drv)}")

for i in range(NUM_RUNS):
    t0 = time.time()
    result = run_model(img, deshadow_prompt, 1024)
    result = run_model(result, appearance_prompt, 1024)
    elapsed = time.time() - t0
    alloc, drv = mem_stats()
    leak = alloc - baseline_alloc
    print(f"  run {i+1:2d}: {elapsed*1000:.0f}ms  alloc={mb(alloc)}  driver={mb(drv)}  leak={mb(leak)}")

cleanup()
final_alloc, final_drv = mem_stats()
print(f"Final:    allocated={mb(final_alloc)}  driver={mb(final_drv)}  total_leak={mb(final_alloc - baseline_alloc)}")


print("\n" + "=" * 60)
print(f"MBD get_mask x{NUM_RUNS}")
print("=" * 60)

cleanup()
baseline_alloc, baseline_drv = mem_stats()
print(f"Baseline: allocated={mb(baseline_alloc)}  driver={mb(baseline_drv)}")

for i in range(NUM_RUNS):
    t0 = time.time()
    mask = get_mask(img)
    elapsed = time.time() - t0
    alloc, drv = mem_stats()
    leak = alloc - baseline_alloc
    print(f"  run {i+1:2d}: {elapsed*1000:.0f}ms  alloc={mb(alloc)}  driver={mb(drv)}  leak={mb(leak)}")

cleanup()
final_alloc, final_drv = mem_stats()
print(f"Final:    allocated={mb(final_alloc)}  driver={mb(final_drv)}  total_leak={mb(final_alloc - baseline_alloc)}")


print("\n" + "=" * 60)
print(f"DEBLUR pipeline x{NUM_RUNS}")
print("=" * 60)

cleanup()
baseline_alloc, baseline_drv = mem_stats()
print(f"Baseline: allocated={mb(baseline_alloc)}  driver={mb(baseline_drv)}")

for i in range(NUM_RUNS):
    t0 = time.time()
    result = run_model(img, deblur_prompt, 1024)
    elapsed = time.time() - t0
    alloc, drv = mem_stats()
    leak = alloc - baseline_alloc
    print(f"  run {i+1:2d}: {elapsed*1000:.0f}ms  alloc={mb(alloc)}  driver={mb(drv)}  leak={mb(leak)}")

cleanup()
final_alloc, final_drv = mem_stats()
print(f"Final:    allocated={mb(final_alloc)}  driver={mb(final_drv)}  total_leak={mb(final_alloc - baseline_alloc)}")


print("\n" + "=" * 60)
print("DIFFERENT IMAGE SIZES")
print("=" * 60)

for size in [(512, 512), (1024, 1024), (1600, 1200), (2048, 1536), (3024, 4032)]:
    w, h = size
    test_img = make_test_image(w, h)
    cleanup()
    before_alloc, before_drv = mem_stats()
    t0 = time.time()
    result = run_model(test_img, appearance_prompt, 1024)
    elapsed = time.time() - t0
    after_alloc, after_drv = mem_stats()
    peak_delta = after_drv - before_drv
    print(f"  {w}x{h}: {elapsed*1000:.0f}ms  driver_delta={mb(peak_delta)}  driver_total={mb(after_drv)}")

cleanup()


print("\n" + "=" * 60)
print("WITHOUT cleanup (simulate old server)")
print("=" * 60)

baseline_alloc, _ = mem_stats()
print(f"Baseline: allocated={mb(baseline_alloc)}")

for i in range(NUM_RUNS):
    in_im_np = np.concatenate((img, appearance_prompt(img)), -1)
    in_im_np, _, _ = stride_integral(in_im_np, 8)
    in_im = torch.from_numpy((in_im_np / 255.0).transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)
        pred = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # NO del, NO cleanup — old behavior
    alloc, drv = mem_stats()
    print(f"  run {i+1:2d} (no cleanup): alloc={mb(alloc)}  driver={mb(drv)}  leak={mb(alloc - baseline_alloc)}")

print("Now cleaning up...")
del in_im, pred
cleanup()
alloc, drv = mem_stats()
print(f"  after cleanup: alloc={mb(alloc)}  driver={mb(drv)}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
