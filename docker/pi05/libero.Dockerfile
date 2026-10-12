# ELASTIC LIBERO-10-R client. Same stack as upstream openpi's examples/libero image, with
# LIBERO-10-R's requirements (the benchmark code itself is mounted at /opt/libero).
#
#   docker build -t elastic_libero -f docker/pi05/libero.Dockerfile .
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    make g++ clang libosmesa6-dev libgl1-mesa-glx libegl1 libglew-dev libglfw3-dev \
    libgles2-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
COPY backends/openpi/examples/libero/requirements.txt /tmp/requirements.txt
COPY backends/libero-10-r/requirements.txt /tmp/requirements-libero.txt
COPY backends/openpi/packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Source builds on Python 3.8 (e.g. evdev) need a setuptools that still supports it.
RUN echo "setuptools<70" > /tmp/build-constraints.txt && \
    uv venv --python 3.8 /.venv && \
    uv pip sync --python /.venv/bin/python --build-constraint /tmp/build-constraints.txt \
      /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml \
      --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# openpi-client from backends/openpi at /app, LIBERO-10-R at /opt/libero, ELASTIC at /elastic (see compose.yml).
ENV PYTHONPATH=/app/packages/openpi-client/src:/opt/libero:/elastic
ENV LIBERO_CONFIG_PATH=/elastic/configs/libero
CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python -m elastic.pi05.serving.libero_client $CLIENT_ARGS"]
