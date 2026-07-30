#!/usr/bin/env python3
"""DocRes appearance enhancement server — production."""

import gc
import os
import sys
import time
import traceback
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from safetensors.torch import load_file
from models.restormer_arch import Restormer
from models.rrdbnet import RRDBNet

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbd"))
from model.deep_lab_model.deeplab import DeepLab

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "geotr"))
from GeoTr import GeoTr
from seg import U2NETP

app = FastAPI()


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)
    print(f"[error] {request.url.path}: {type(exc).__name__}: {exc}\n{tb_str}")
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "traceback": tb_str,
        }
    )

API_KEY = os.environ.get("DOCRES_API_KEY", "")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30  # requests per window per IP
rate_limit_store: dict[str, list[float]] = {}
rate_limit_lock = threading.Lock()

gpu_lock = threading.Semaphore(1)
gpu_queue_count = 0
gpu_queue_lock = threading.Lock()
GPU_MAX_QUEUE = 15


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

MAX_DIM = 1024
ALLOWED_RESOLUTIONS = {512, 768, 1024, 1600, 2048}
DEFAULT_RESOLUTION = 2048
ESRGAN_MIN_DIM = 1500
model = None
mbd_model = None
geotr_model = None
esrgan_model = None


GPU_COOLDOWN = 0.15

def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(GPU_COOLDOWN)


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

    try:
        with torch.no_grad():
            pred = model(in_im)
            pred = torch.clamp(pred, 0, 1)
            pred = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    finally:
        del in_im
        _cleanup_gpu()
    return pred[pad_h:, pad_w:]


def sharpen(img, amount=0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


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
        _cleanup_gpu()
    kernel = np.ones((3, 3))
    mask = cv2.dilate(mask, kernel, iterations=15)
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

    try:
        with torch.no_grad():
            model.float()
            pred = model(in_im)
            pred = pred[0][:2].permute(1, 2, 0).cpu().numpy()
            pred = pred + base_coord
    finally:
        model.half()
        del in_im, img_t, prompt_t
        _cleanup_gpu()

    for _ in range(15):
        pred = cv2.blur(pred, (3, 3), borderType=cv2.BORDER_REPLICATE)
    margin = 0.04
    pred = pred * (1 - 2 * margin) + margin
    pred = cv2.resize(pred, (w, h)) * (w, h)
    pred = pred.astype(np.float32)
    out = cv2.remap(img_bgr, pred[:, :, 0], pred[:, :, 1], cv2.INTER_LINEAR)
    out = cv2.resize(out, (w, h))
    return out


def deblur_prompt(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    hf = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)


class GeoTr_Seg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x, expand=0.0):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
        msk = (msk > 0.5).to(x.dtype)
        x = msk * x
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        if expand > 0:
            bm = bm * (1 + 2 * expand)
        return bm


def geotr_dewarp(img_bgr, expand=0.02):
    dtype = next(geotr_model.parameters()).dtype
    im_ori = img_bgr[:, :, ::-1] / 255.0
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
    im = torch.from_numpy(im).unsqueeze(0).to(DEVICE, dtype=dtype)
    try:
        with torch.no_grad():
            bm = geotr_model(im, expand=expand).float().cpu()
            bm0 = cv2.blur(cv2.resize(bm[0, 0].numpy(), (w, h)), (3, 3))
            bm1 = cv2.blur(cv2.resize(bm[0, 1].numpy(), (w, h)), (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).to(DEVICE)
            src = torch.from_numpy(im_ori.copy()).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            out = F.grid_sample(src, lbl, align_corners=True, padding_mode="border")
            result = ((out[0] * 255).permute(1, 2, 0).cpu().numpy())[:, :, ::-1].astype(np.uint8)
    finally:
        del im
        _cleanup_gpu()
    return result


def esrgan_upscale(img_bgr):
    dtype = next(esrgan_model.parameters()).dtype
    img = torch.from_numpy(img_bgr[:, :, ::-1].copy().transpose(2, 0, 1)).unsqueeze(0) / 255.0
    img = img.to(DEVICE, dtype=dtype)
    try:
        with torch.no_grad():
            out = esrgan_model(img)
        result = (out[0].float().clamp(0, 1).cpu().numpy() * 255).transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
    finally:
        del img, out
        _cleanup_gpu()
    return result


def _load_prefixed(module, path, prefix_len):
    md = module.state_dict()
    pd = torch.load(path, map_location="cpu")
    pd = {k[prefix_len:]: v for k, v in pd.items() if k[prefix_len:] in md}
    md.update(pd)
    module.load_state_dict(md)


@app.on_event("startup")
def load_model():
    global model, mbd_model, geotr_model, esrgan_model
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
    mbd_model = mbd_model.half().to(DEVICE)

    geotr_model = GeoTr_Seg()
    _load_prefixed(geotr_model.msk, "checkpoints/seg.pth", 6)
    _load_prefixed(geotr_model.GeoTr, "checkpoints/geotr.pth", 7)
    geotr_model.eval()
    geotr_model = geotr_model.float().to(DEVICE)

    esrgan_model = RRDBNet(scale=2)
    esr_sd = torch.load("checkpoints/esrgan_x2.pth", map_location="cpu")
    esr_sd = esr_sd.get("params_ema", esr_sd.get("params", esr_sd))
    esrgan_model.load_state_dict(esr_sd, strict=True)
    esrgan_model.eval()
    esrgan_model = esrgan_model.to(DEVICE)
    if DEVICE.type == "cuda":
        esrgan_model = esrgan_model.half()

    print(f"[server] Models loaded in {time.time()-t0:.1f}s on {DEVICE}")


def _process_enhance(data, max_dim=MAX_DIM):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_model(img_bgr, deshadow_prompt, max_dim)
    result = run_model(result, appearance_prompt, max_dim)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[enhance] {w}x{h} @ {max_dim} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_dewarp(data, max_dim=MAX_DIM):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = geotr_dewarp(img_bgr)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[dewarp] {w}x{h} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_full(data, max_dim=MAX_DIM):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    if max(h, w) < ESRGAN_MIN_DIM:
        img_bgr = esrgan_upscale(img_bgr)
    img_bgr = geotr_dewarp(img_bgr)
    result = run_model(img_bgr, deshadow_prompt, max_dim)
    result = run_model(result, appearance_prompt, max_dim)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[full] {w}x{h} @ {max_dim} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _process_deblur(data, max_dim=MAX_DIM):
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    t = time.time()
    result = run_model(img_bgr, deblur_prompt, max_dim)
    result = sharpen(result)
    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[deblur] {w}x{h} @ {max_dim} -> {(time.time()-t)*1000:.0f}ms")
    return buf


def _parse_resolution(request):
    raw = request.query_params.get("resolution")
    if raw is None:
        return DEFAULT_RESOLUTION
    try:
        val = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resolution")
    if val not in ALLOWED_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
    return val


def _run_pipeline(request, data, process_fn, media_type, max_dim=MAX_DIM):
    global gpu_queue_count
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
    with gpu_queue_lock:
        if gpu_queue_count >= GPU_MAX_QUEUE:
            retry_after = gpu_queue_count * 2
            return JSONResponse(
                status_code=429,
                content={"error": "Server busy, try again later"},
                headers={"Retry-After": str(retry_after)},
            )
        gpu_queue_count += 1
    try:
        gpu_lock.acquire()
        try:
            for attempt in range(2):
                try:
                    buf = process_fn(data, max_dim)
                    break
                except Exception as e:
                    _cleanup_gpu()
                    if attempt == 0:
                        print(f"[retry] {type(e).__name__}: {e}, retrying...")
                        continue
                    raise
        finally:
            _cleanup_gpu()
            gpu_lock.release()
    finally:
        with gpu_queue_lock:
            gpu_queue_count -= 1
    if buf is None:
        return Response(content="bad image", status_code=400)
    return Response(content=buf.tobytes(), media_type=media_type)


@app.post("/enhance/quality")
def enhance_quality(request: Request, file: UploadFile = File(...)):
    max_dim = _parse_resolution(request)
    data = file.file.read()
    return _run_pipeline(request, data, _process_enhance, "image/jpeg", max_dim)


@app.post("/dewarp")
def dewarp(request: Request, file: UploadFile = File(...)):
    data = file.file.read()
    return _run_pipeline(request, data, _process_dewarp, "image/jpeg")


@app.post("/full")
def full_pipeline(request: Request, file: UploadFile = File(...)):
    max_dim = _parse_resolution(request)
    data = file.file.read()
    return _run_pipeline(request, data, _process_full, "image/jpeg", max_dim)


@app.post("/deblur")
def deblur(request: Request, file: UploadFile = File(...)):
    max_dim = _parse_resolution(request)
    data = file.file.read()
    return _run_pipeline(request, data, _process_deblur, "image/jpeg", max_dim)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gpu_busy": gpu_lock._value == 0,
    }
