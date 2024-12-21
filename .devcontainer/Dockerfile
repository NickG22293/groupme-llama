FROM mcr.microsoft.com/vscode/devcontainers/python:3
USER root

# install yq 
RUN sudo wget https://github.com/mikefarah/yq/releases/download/v4.30.5/yq_linux_amd64.tar.gz -O - |\
  sudo tar xz && mv yq_linux_amd64 /usr/bin/yq

# ML and llama libraries for python 
RUN pip install torch transformers peft