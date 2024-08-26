# Use NVIDIA CUDA 11.1 base image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1

# Install Mambaforge
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" -O mambaforge.sh && \
    bash mambaforge.sh -b -p /opt/conda && \
    rm mambaforge.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize mamba for shell interaction
RUN mamba init bash

# Create a new conda environment and install packages
RUN mamba create -n flowdiffuser-docker python=3.8 && \
    mamba run -n flowdiffuser-docker mamba install -y \
    pytorch=1.9.0 \
    torchvision=0.10.0 \
    numpy=1.19.5 \
    opencv \
    timm=0.6.12 \
    scipy \
    matplotlib \
    tqdm \
    tensorboard \
    -c conda-forge

# Set up the entrypoint to activate the environment
#SHELL ["/bin/bash", "-c"]
#ENTRYPOINT ["mamba", "run", "-n", "flowdiffuser-docker"]
#CMD ["/bin/bash"]

# Create an activation script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/mamba.sh\n\
mamba activate flowdiffuser-docker\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh