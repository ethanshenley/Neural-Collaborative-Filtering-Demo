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
    google-cloud-bigquery[bqstorage,pandas] \
    pandas-gbq>=0.20.0 \ 
    pyyaml>=6.0.1 \
    pandas \
    numpy \
    tqdm \
    db-dtypes

# Verify CUDA setup
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Copy configuration first to verify it
COPY config/ /app/config/

# Verify config.yaml exists and can be loaded
RUN python -c "import yaml, os; \
    config_path = '/app/config/config.yaml'; \
    assert os.path.exists(config_path), f'Config not found at {config_path}'; \
    config = yaml.safe_load(open(config_path)); \
    assert 'data' in config, 'data section missing in config'; \
    assert 'model' in config, 'model section missing in config'; \
    assert 'gcp' in config, 'gcp section missing in config'; \
    print('Config validation successful')"

# Copy remaining application files
COPY setup.py requirements.txt ./
COPY src/ /app/src/

# Install the package in editable mode
RUN pip install -e .

# Verify complete setup
COPY verify_setup.py ./
RUN python verify_setup.py

# Verify train.py and check structure
RUN python -c "import os; \
    assert os.path.exists('/app/src/train.py'), 'train.py not found'; \
    from src.utils.config import ConfigLoader; \
    from src.model.architecture import AdvancedNCF; \
    print('All critical modules imported successfully')"

# Set default command
CMD ["python", "-m", "src.train"]