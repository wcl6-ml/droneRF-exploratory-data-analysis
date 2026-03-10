FROM nvcr.io/nvidia/pytorch:25.10-py3

RUN apt-get update && apt-get install -y \
    tree vim bash-completion sudo curl git \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
RUN userdel ubuntu && \
    useradd -m -s /bin/bash -u 1000 $USERNAME \
    && mkdir -p /home/vscode/.local/state/mise \
               /home/vscode/.local/share/mise \
               /home/vscode/.cache/ \
    && chown -R vscode:vscode /home/vscode/.local \
    && chown -R vscode:vscode /home/vscode/.cache \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh
RUN echo 'eval "$(mise activate bash)"' >> /home/vscode/.bashrc

USER $USERNAME
WORKDIR /workspace
