FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

#-------------------------------------------------
# 1) System packages
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
# 2) User setup
#-------------------------------------------------
USER vscode

#-------------------------------------------------
# 3) Miniconda installation
#-------------------------------------------------
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/home/vscode/miniconda3/bin:${PATH}"

#-------------------------------------------------
# 4) Copy environment file and create environment
#-------------------------------------------------
WORKDIR /app
COPY environment.yml /tmp/environment.yml

# Remove old environment if it exists, then create a fresh one
RUN conda env remove -n llava_yaml -y || true && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy

#-------------------------------------------------
# 5) Install dotnet (optional, if you really need it)
#-------------------------------------------------
RUN cd /tmp && \
    wget https://dot.net/v1/dotnet-install.sh && \
    chmod +x dotnet-install.sh && \
    ./dotnet-install.sh --channel 7.0 && \
    ./dotnet-install.sh --channel 3.1 && \
    rm ./dotnet-install.sh

#-------------------------------------------------
# 6) Copy in your project
#-------------------------------------------------
COPY . /app

# Switch to root just to chmod your script
USER root
RUN chmod +x ./scripts/v1_5/anyres_pretrain.sh

# Switch back to vscode for final
USER vscode

#-------------------------------------------------
# 7) Entry point
#-------------------------------------------------
# Instead of "source activate", which can be tricky in non-interactive shells,
# use 'conda run' to ensure the environment is active when your script runs.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "llava_yaml", "bash", "./scripts/v1_5/anyres_pretrain.sh"]
