# ============================================================
# Dockerfile — Gemma 4 inference + eval (llama-cpp + HF)
# Base : PyTorch + CUDA 12.1 runtime (compatible RunPod GPU)
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cache HF redirigé vers un volume montable (évite de saturer le container disk)
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    CODECARBON_LOG_LEVEL=error

# ca-certificates suffit : on n'a plus besoin de build-essential/cmake
# car on installe le wheel CUDA pre-compile de llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Wheel CUDA 12.1 pre-compile -> pas de compilation, ~10 min et plusieurs Go gagnes
RUN pip install --upgrade pip && \
    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /cache/huggingface

CMD ["python", "test_infer_gemma4.py"]
