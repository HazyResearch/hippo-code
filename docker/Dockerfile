# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.5.0-cuda10.2-ubuntu18.04/Dockerfile
FROM nvidia/cuda:10.2-runtime-ubuntu18.04
ARG PYTHON_VERSION=3.8

# Need rsync for ray (and ssh for ray with docker)
# htop to ssh into docker container and monitor jobs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    sudo \
    rsync \
    openssh-client \
    htop \
    git \
    && rm -rf /var/lib/apt/lists/*

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#     && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user

# Install conda, python
ENV PATH /home/user/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/conda \
    && rm ~/miniconda.sh \
    && conda install -y python=$PYTHON_VERSION \
    && conda clean -ya

# Pytorch, scipy
RUN conda install -y -c pytorch cudatoolkit=10.2 pytorch=1.7.1 torchvision \
    && conda install -y scipy \
    && conda clean -ya

# Other libraries
# wanbd>=0.10.0 tries to read from ~/.config, and that causes permission error on dawn
RUN pip install pytorch-lightning==1.1.2 ray[tune]==1.1.0 hydra-core==1.0.4 wandb==0.9.7 munch scikit-learn sktime==0.5.1 \
    && rm -rf /home/user/.cache/pip
