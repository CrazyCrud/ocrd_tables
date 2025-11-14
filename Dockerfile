ARG DOCKER_BASE_IMAGE=ocrd/core-cuda-torch:2024
FROM ${DOCKER_BASE_IMAGE}

ARG VCS_REF
ARG BUILD_DATE

LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.opencontainers.image.title="ocrd-tables" \
    org.opencontainers.image.description="OCR-D processor: fuse YOLO columns and textlines into table cells via DBSCAN" \
    org.opencontainers.image.source="https://github.com/CrazyCrud/ocrd-tables" \
    org.opencontainers.image.documentation="https://github.com/CrazyCrud/ocrd-tables" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for OCR-D"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Make OCR-D resource caches predictable & writable
ENV XDG_DATA_HOME=/usr/local/share
ENV XDG_CONFIG_HOME=/usr/local/share/ocrd-resources

# Workdir for building the wheel / installing
WORKDIR /build/ocrd_tables

# Copy project into image
# (Assumes the repo layout described earlier, with pyproject.toml and ocrd-tool.json at repo root)
COPY . .

# Ensure ocrd-tool.json is present at repo root (uncomment if you keep it inside the package dir)
# COPY src/ocrd_tables/ocrd-tool.json ./ocrd-tool.json

# System deps (kept minimal; wheels exist for shapely/sklearn, but we keep a few libs for OpenCV/NumPy ABI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Prepackage OCR-D tool metadata (optional but handy for ocrd-all style discovery)
# (This expects ocrd is already in the base image, which it is.)
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json && \
    ocrd ocrd-tool ocrd-tool.json dump-module-dirs > $(dirname $(ocrd bashlib filename))/ocrd-all-module-dir.json

# Install Python deps and the processor itself
# (Relies on pyproject.toml to declare: ocrd, ocrd-models, shapely, numpy, scikit-learn, etc.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Clean build context to reduce final size
RUN rm -rf /build/ocrd_tables

# Runtime working directory & volume
WORKDIR /data
VOLUME ["/data"]

# Default help (override in `docker run` with your actual CLI call)
# Example run:
# docker run --gpus all --rm -v $PWD:/data image \
#   ocrd-tables -I OCR-D-SEG-COLUMN,OCR-D-SEG-TEXTLINE-TABLE -O OCR-D-TABLE-CELLS
CMD ["ocrd-tables", "--help"]
