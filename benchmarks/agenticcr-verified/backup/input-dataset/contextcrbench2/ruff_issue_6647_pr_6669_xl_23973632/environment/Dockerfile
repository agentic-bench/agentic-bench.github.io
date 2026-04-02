FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /env_setup
COPY setup_repo.sh /env_setup/setup_repo.sh
COPY code_change_context.json /workspace/code_change_context.json
COPY ground_truth.json /env_setup/ground_truth.json
COPY dataset.json /env_setup/dataset.json
RUN chmod +x /env_setup/setup_repo.sh

# *** Run setup at build time ***
RUN /env_setup/setup_repo.sh