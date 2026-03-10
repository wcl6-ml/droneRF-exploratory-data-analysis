FROM nvcr.io/nvidia/pytorch:25.10-py3

RUN apt-get update && apt-get install -y \
    tree vim bash-completion sudo curl git \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
RUN userdel ubuntu && \
    useradd -m -s /bin/bash -u 1000 $USERNAME \
    && mkdir -p /home/vscode/.cache/ \
    && chown -R vscode:vscode /home/vscode/.cache \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


USER $USERNAME
WORKDIR /workspace
