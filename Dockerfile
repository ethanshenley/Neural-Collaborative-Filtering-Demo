FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-7 \
    && rm -rf /var/lib/apt/lists/*

# Install cloud dependencies
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Install Python dependencies
RUN pip install --no-cache-dir \
    torchrec>=0.6.0 \
    google-cloud-bigquery>=3.17.1 \
    google-cloud-storage>=2.14.0 \
    pyyaml>=6.0.1 \
    pandas \
    numpy \
    fbgemm-gpu

# Set working directory
WORKDIR /app

# Copy project files
COPY src/ /app/src/
COPY config/ /app/config/
COPY setup.py /app/
COPY requirements.txt /app/

# Install the package
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app