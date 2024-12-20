# Use PyTorch base image with CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA repository and install CUDA libraries
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libnccl2 \
    libnccl-dev \
    cuda-libraries-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and ecosystem packages
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchrec==0.8.0 \
    fbgemm-gpu==0.8.0 \
    torchmetrics==1.0.3 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    google-cloud-bigquery>=3.17.1 \
    google-cloud-storage>=2.14.0 \
    pyyaml>=6.0.1 \
    pandas \
    numpy \
    tqdm

# Verify CUDA setup
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Add and verify setup script
COPY verify_setup.py /verify_setup.py
RUN python /verify_setup.py

# Set working directory
WORKDIR /app

# Copy application files
COPY setup.py /app/
COPY requirements.txt /app/
COPY src/ /app/src/
COPY config/ /app/config/

# Install the package in editable mode
RUN pip install -e .

# Verify `train.py` exists and inspect its first 20 lines
RUN echo "Verifying train.py version:" && head -n 20 /app/src/train.py

# Entry point (optional, for local testing)
CMD ["python", "-m", "src.train"]
