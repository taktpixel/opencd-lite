# Development / test image for opencd-lite (CPU-only PyTorch).
#
# The project is installed editable against /workspace, which
# docker-compose bind-mounts to the repository checkout, so source
# changes do not require a rebuild — only dependency changes do.

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install CPU wheels first so the (large) CUDA distribution is never pulled.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install the package and all development dependencies.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[dev]"

COPY . .

CMD ["pytest"]
