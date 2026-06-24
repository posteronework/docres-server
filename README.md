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
docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name docres docres
```

### 2b. Run without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API

### Health check

```
GET /health
```

### Endpoints

```
POST /full              — dewarp + deshadow + appearance + sharpen (full pipeline)
POST /enhance/quality   — deshadow + appearance + sharpen
POST /dewarp            — document dewarping only
POST /deblur            — deblurring + sharpen
```

All endpoints accept multipart form data with a `file` field (JPEG/PNG).
Return enhanced JPEG image.

Example:

```bash
curl -X POST http://localhost:8000/enhance/quality -F "file=@document.jpg" -o enhanced.jpg
```

## Hardware requirements

- **GPU VRAM**: ~2.2 GB (Restormer 183MB + MBD 680MB + inference buffers)
- **Recommended**: NVIDIA RTX 4090
- **Expected latency**: /enhance/quality ~1s, /full ~1.2s on RTX 4090

Also works on Apple Silicon (MPS) and CPU (slower).

## Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| DocRes (Restormer) | `checkpoints/docres.pkl` | 183 MB | Appearance, deshadow, deblur, dewarping flow |
| MBD (DeepLab ResNet) | `checkpoints/mbd.pkl` | 680 MB | Document mask for dewarping |

## Project structure

```
├── server.py              # FastAPI server
├── models/
│   └── restormer_arch.py  # Restormer architecture
├── mbd/                   # MBD model code (DeepLab)
├── checkpoints/
│   ├── docres.pkl         # Restormer weights (Git LFS)
│   └── mbd.pkl            # MBD weights (Git LFS)
├── requirements.txt
└── Dockerfile
```
