# DocRes Server

Document appearance enhancement server based on [Restormer](https://arxiv.org/abs/2111.09881).

Single-pass inference at up to 1600px resolution with appearance prompt + sharpening.

## Setup

### 1. Download model weights

Download `docres.pkl` (175MB) and place it in `checkpoints/`:

```
mkdir -p checkpoints
# Get the file from the team or internal storage
cp /path/to/docres.pkl checkpoints/
```

### 2a. Run with Docker (recommended)

```bash
docker build -t docres .
docker run -p 8000:8000 --gpus all docres
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

### Enhance document image

```
POST /v1  — 1 pass, 1024px (fastest)
POST /v2  — 1 pass, 1600px (recommended)
POST /v3  — 2 passes, 1024px
POST /v4  — 2 passes, 1600px (highest quality)
```

All endpoints accept multipart form data with a `file` field (JPEG/PNG).
Return enhanced JPEG image.

Example:

```bash
curl -X POST http://localhost:8000/v2 -F "file=@document.jpg" -o enhanced.jpg
```

## Hardware requirements

- **GPU VRAM**: ~2 GB (model weights + inference)
- **Recommended**: NVIDIA A10 or RTX 4090
- **Expected latency** (1 pass, 1600px): ~1-2s on A10, ~1.5-2s on 4090

Also works on Apple Silicon (MPS) and CPU (slower).

## Project structure

```
├── server.py           # FastAPI server
├── models/
│   └── restormer_arch.py  # Model architecture
├── checkpoints/
│   └── docres.pkl      # Model weights (not in git, see Setup)
├── requirements.txt
└── Dockerfile
```
