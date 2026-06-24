FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG TARGETPLATFORM
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY mbd/ mbd/
COPY checkpoints/ checkpoints/
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
