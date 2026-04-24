# ============================================================
# Dockerfile — Gemma 4 inference + eval (llama-cpp + HF)
# Base : PyTorch + CUDA 12.1 (compatible RunPod GPU)
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cache HF redirigé vers un volume montable (évite de saturer le container disk)
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    CODECARBON_LOG_LEVEL=error

# Outils systeme necessaires pour compiler llama-cpp-python avec CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les deps Python (on copie requirements.txt d'abord pour profiter du cache Docker)
COPY requirements.txt .

# llama-cpp-python compile avec support CUDA (GGML_CUDA=on)
ENV CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code du projet
COPY . .

# Creer le dossier cache (volume a monter au runtime)
RUN mkdir -p /cache/huggingface

# Par defaut on lance le script d'eval — modifie selon besoin
CMD ["python", "test_infer_gemma4.py"]
