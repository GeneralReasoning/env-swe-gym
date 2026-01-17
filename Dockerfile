FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y \
    software-properties-common \
    docker.io \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    git \
    git-lfs \
    wget \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
RUN uv venv --python 3.11

# Install dependencies
COPY construct /app/construct
COPY matrix /app/matrix
COPY environments/pyproject.toml /app/environments/pyproject.toml
COPY environments/environments/swegym/ /app/environments/environments/swegym

RUN uv pip install -e /app/construct/client && \
    uv pip install -e /app/matrix && \
    uv pip install -e /app/environments

# Install application
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -r /app/environments/environments/swegym/requirements.txt

EXPOSE 8080
CMD ["uv", "run", "python", "/app/environments/environments/swegym/server.py"]
