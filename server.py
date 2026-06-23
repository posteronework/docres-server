#!/usr/bin/env python3
"""DocRes appearance enhancement server — production."""

import time
import cv2
import numpy as np
import torch
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from models.restormer_arch import Restormer

app = FastAPI()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MAX_DIM = 1600
model = None


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    return new_state_dict


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

    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return pred[pad_h:, pad_w:]


def sharpen(img, amount=0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


@app.on_event("startup")
def load_model():
    global model
    t0 = time.time()
    model = Restormer(
        inp_channels=6, out_channels=3, dim=48,
        num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=True,
    )
    state = convert_state_dict(
        torch.load("checkpoints/docres.pkl", map_location="cpu")["model_state"]
    )
    model.load_state_dict(state)
    model.eval()
    model = model.half().to(DEVICE)
    print(f"[server] Model loaded in {time.time()-t0:.1f}s on {DEVICE}")


@app.post("/enhance/quality")
async def enhance_quality(file: UploadFile = File(...)):
    t0 = time.time()

    data = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return Response(content="bad image", status_code=400)

    h, w = img_bgr.shape[:2]
    t_infer = time.time()
    result = run_model(img_bgr, deshadow_prompt, MAX_DIM)
    result = run_model(result, appearance_prompt, MAX_DIM)
    result = sharpen(result)
    infer_ms = (time.time() - t_infer) * 1000

    _, buf = cv2.imencode(".png", result)
    total_ms = (time.time() - t0) * 1000
    print(f"[server] {w}x{h} -> infer {infer_ms:.0f}ms, total {total_ms:.0f}ms")

    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}
