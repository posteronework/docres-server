# DocRes Server

Document enhancement server based on [DocRes](https://arxiv.org/abs/2405.04408) (Restormer) + [DeepLab](https://arxiv.org/abs/1706.05587) (MBD).

Supports appearance enhancement, deshadowing, dewarping, and deblurring at up to 1600px resolution.

## Setup

### 1. Clone (includes model weights via Git LFS)

```bash
git lfs install
git clone git@github.com:posteronework/docres-server.git
cd docres-server
```

### 2a. Run with Docker (recommended)

```bash
docker build -t docres .
docker run -d --gpus all -p 8000:8000 \
  -e DOCRES_API_KEY="your_secret_key" \
  --restart unless-stopped --name docres docres
```

### 2b. Run without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
DOCRES_API_KEY="your_secret_key" uvicorn server:app --host 0.0.0.0 --port 8000
```

## API

All POST endpoints require `Authorization: Bearer <API_KEY>` header.

### Health check

```
GET /health
```

Returns `{"status": "ok", "device": "cuda", "gpu_busy": false}`

### Endpoints

```
POST /full              — dewarp + deshadow + appearance + sharpen (full pipeline)
POST /enhance/quality   — deshadow + appearance + sharpen
POST /dewarp            — document dewarping only
POST /deblur            — deblurring + sharpen
```

All endpoints accept multipart form data with a `file` field (JPEG/PNG).
Return enhanced JPEG (quality 95%) image. Max file size: 20MB.

Example:

```bash
curl -X POST http://localhost:8000/enhance/quality \
  -H "Authorization: Bearer your_secret_key" \
  -F "file=@document.jpg" -o enhanced.jpg
```

## Protection

- **API key** — via `DOCRES_API_KEY` env variable, checked in `Authorization: Bearer` header
- **Rate limiting** — 30 requests/min per IP
- **File size limit** — 20MB (checked via Content-Length before upload)
- **GPU queue** — `Semaphore(1)` ensures sequential GPU access, non-blocking event loop

## Hardware requirements

- **GPU VRAM**: ~1.2 GB (Restormer 58MB + MBD 227MB + inference buffers)
- **Recommended**: NVIDIA RTX 4090
- **Expected latency**: /enhance/quality ~1s, /full ~1.2s on RTX 4090

Also works on Apple Silicon (MPS) and CPU (slower).

## Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| DocRes (Restormer) | `checkpoints/docres.safetensors` | 58 MB | Appearance, deshadow, deblur, dewarping flow |
| MBD (DeepLab ResNet) | `checkpoints/mbd.safetensors` | 227 MB | Document mask for dewarping |

## Project structure

```
├── server.py              # FastAPI server
├── models/
│   └── restormer_arch.py  # Restormer architecture
├── mbd/                   # MBD model code (DeepLab)
├── checkpoints/
│   ├── docres.safetensors # Restormer weights (Git LFS)
│   └── mbd.safetensors    # MBD weights (Git LFS)
├── requirements.txt
├── Dockerfile
└── .dockerignore
```
