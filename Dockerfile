FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

#-------------------------------------------------
# 1) Install System Packages
#-------------------------------------------------
RUN apt update && \
    apt install -yq \
        ffmpeg \
        dkms \
        build-essential \
        jq \
        jp \
        tree \
        tldr \
        curl \
        wget \
        git

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install -yq git-lfs && \
    git lfs install

#-------------------------------------------------
# 2) Working Directory & Code
#    Put code in /app instead of /storage
#-------------------------------------------------
WORKDIR /app

# Copy the environment file *before* the rest of your code
# so that Docker can cache the env creation if environment.yml doesn't change often
COPY environment.yml /tmp/environment.yml

# Then copy your entire codebase into /app
COPY . /app

#-------------------------------------------------
# 3) Install Miniconda
#-------------------------------------------------
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/vscode/miniconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

# Ensure conda is on PATH in all subsequent commands
ENV PATH="/home/vscode/miniconda3/bin:$PATH"

#-------------------------------------------------
# 4) Create Conda Environment
#-------------------------------------------------
RUN conda env remove -n llava_yaml -y || true && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Optionally install flash-attn if CUDA is present
RUN if command -v nvcc >/dev/null 2>&1; then \
    echo "✅ CUDA detected — installing flash-attn..." && \
    conda run -n llava_yaml pip install flash-attn --no-build-isolation --no-cache-dir; \
else \
    echo "⚠️  Skipping flash-attn install — CUDA not found"; \
fi

#-------------------------------------------------
# 5) (Optional) Dotnet Installation
#-------------------------------------------------
RUN cd /tmp && \
    wget https://dot.net/v1/dotnet-install.sh && \
    chmod +x dotnet-install.sh && \
    ./dotnet-install.sh --channel 7.0 && \
    ./dotnet-install.sh --channel 3.1 && \
    rm ./dotnet-install.sh

#-------------------------------------------------
# 6) Make Your Training Script Executable
#-------------------------------------------------
RUN chmod +x /app/scripts/v1_5/anyres_pretrain.sh

#-------------------------------------------------
# 7) Entry Point
#    Use 'conda run' to ensure the llava_yaml env is active
#-------------------------------------------------
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "llava_yaml", "bash", "/app/scripts/v1_5/anyres_pretrain.sh"]
