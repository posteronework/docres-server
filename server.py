#!/usr/bin/env python3
"""DocRes appearance enhancement server — production."""

import asyncio
import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from safetensors.torch import load_file
from models.restormer_arch import Restormer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbd"))
from model.deep_lab_model.deeplab import DeepLab

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[error] {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

API_KEY = os.environ.get("DOCRES_API_KEY", "")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30  # requests per window per IP
rate_limit_store: dict[str, list[float]] = {}
rate_limit_lock = threading.Lock()

gpu_semaphore = asyncio.Semaphore(1)


def check_auth(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    with rate_limit_lock:
        timestamps = rate_limit_store.get(ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            raise HTTPException(status_code=429, detail="Too many requests")
        timestamps.append(now)
        rate_limit_store[ip] = timestamps
        stale = [k for k, v in rate_limit_store.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW]
        for k in stale:
            del rate_limit_store[k]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MAX_DIM = 1600
model = None
mbd_model = None


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


def run_model(img_bgr, prompt_fn, max_dim):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    prompt = prompt_fn(img_bgr)
    in_im = np.concatenate((img_bgr, prompt), -1)
    in_im, pad_h, pad_w = stride_integral(in_im, 8)

    in_im = torch.from_numpy((in_im / 255.0).transpose(2, 0, 1)).unsqueeze(0).half().to(DEVICE)

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)
        pred = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return pred[pad_h:, pad_w:]


def sharpen(img, amount=0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def get_mask(img_bgr):
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (448, 448))
    img = cv2.GaussianBlur(img, (15, 15), 0, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy((img.astype(float) / 255.0).transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = mbd_model(img_t)
        mask = pred[:, 0, :, :].unsqueeze(1)
        mask = F.interpolate(mask, (h, w))
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3))
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask[mask > 100] = 255
    mask[mask < 100] = 0
    return mask


def get_base_coord(h, w):
    c0 = np.tile(np.arange(h).reshape(h, 1), (1, w)).astype(np.float32)
    c1 = np.tile(np.arange(w).reshape(1, w), (h, 1)).astype(np.float32)
    return np.concatenate((np.expand_dims(c1, -1), np.expand_dims(c0, -1)), -1)


def run_dewarp(img_bgr):
    INPUT_SIZE = 256
    h, w = img_bgr.shape[:2]
    mask = get_mask(img_bgr)

    img_masked = img_bgr.copy()
    img_masked[mask == 0] = 0
    img_small = cv2.resize(img_masked, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    base_coord = get_base_coord(INPUT_SIZE, INPUT_SIZE) / INPUT_SIZE
    mask_small = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    prompt = np.concatenate((base_coord, np.expand_dims(mask_small, -1)), -1)
    prompt_t = torch.from_numpy(prompt.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    in_im = torch.cat((img_t, prompt_t), dim=1)

    with torch.no_grad():
        model.float()
        pred = model(in_im)
        pred = pred[0][:2].permute(1, 2, 0).cpu().numpy()
        pred = pred + base_coord
        model.half()

    for _ in range(15):
        pred = cv2.blur(pred, (3, 3), borderType=cv2.BORDER_REPLICATE)
    pred = cv2.resize(pred, (w, h)) * (w, h)
    pred = pred.astype(np.float32)
    out = cv2.remap(img_bgr, pred[:, :, 0], pred[:, :, 1], cv2.INTER_LINEAR)
    return out


def deblur_prompt(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    hf = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)


@app.on_event("startup")
def load_model():
    global model, mbd_model
    t0 = time.time()
    model = Restormer(
        inp_channels=6, out_channels=3, dim=48,
        num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=True,
    )
    model.load_state_dict(load_file("checkpoints/docres.safetensors"))
    model.eval()
    model = model.half().to(DEVICE)

    mbd_model = DeepLab(num_classes=1, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
    mbd_model.load_state_dict(load_file("checkpoints/mbd.safetensors"))
    mbd_model.eval()
    mbd_model = mbd_model.to(DEVICE)

    print(f"[server] Models loaded in {time.time()-t0:.1f}s on {DEVICE}")


def _process_enhance(data):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_model(img_bgr, deshadow_prompt, MAX_DIM)
    result = run_model(result, appearance_prompt, MAX_DIM)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[enhance] {w}x{h} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_dewarp(data):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_dewarp(img_bgr)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[dewarp] {w}x{h} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_full(data):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_dewarp(img_bgr)
    result = run_model(result, deshadow_prompt, MAX_DIM)
    result = run_model(result, appearance_prompt, MAX_DIM)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[full] {w}x{h} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_deblur(data):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_model(img_bgr, deblur_prompt, MAX_DIM)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[deblur] {w}x{h} -> {(time.time()-t)*1000:.0f}ms")
    return buf


async def _run_pipeline(request, data, process_fn, media_type):
    check_auth(request)
    check_rate_limit(request)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
        except ValueError:
            pass
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    loop = asyncio.get_event_loop()
    async with gpu_semaphore:
        buf = await loop.run_in_executor(None, process_fn, data)
    if buf is None:
        return Response(content="bad image", status_code=400)
    return Response(content=buf.tobytes(), media_type=media_type)


@app.post("/enhance/quality")
async def enhance_quality(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    return await _run_pipeline(request, data, _process_enhance, "image/jpeg")


@app.post("/dewarp")
async def dewarp(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    return await _run_pipeline(request, data, _process_dewarp, "image/jpeg")


@app.post("/full")
async def full_pipeline(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    return await _run_pipeline(request, data, _process_full, "image/jpeg")


@app.post("/deblur")
async def deblur(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    return await _run_pipeline(request, data, _process_deblur, "image/jpeg")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gpu_busy": gpu_semaphore._value == 0,
    }
