# ELASTIC pi0.5 policy server: upstream openpi's server image plus V-GPS's TensorFlow stack.
#
#   docker build -t openpi_server -f backends/openpi/scripts/docker/serve_policy.Dockerfile backends/openpi
#   docker build -t elastic_pi05_server - < docker/pi05/server.Dockerfile
ARG BASE=openpi_server
FROM ${BASE}

# jaxrl_m (V-GPS) dependencies, pinned to the versions V-GPS was trained with (TF 2.15).
# --no-deps keeps openpi's resolved JAX/flax versions untouched.
RUN uv pip install --python /.venv/bin/python --no-deps \
    "tensorflow-cpu==2.15.0" \
    "tensorflow-hub==0.16.1" \
    "tensorflow-text==2.15.0" \
    "tensorflow-probability==0.23.0" \
    "distrax==0.1.5" && \
    uv pip install --python /.venv/bin/python --no-deps \
    "astunparse>=1.6.3" "gast==0.5.4" "google-pasta>=0.2.0" "h5py>=3.10.0" "libclang>=16.0.6" \
    "opt-einsum>=3.3.0" "protobuf>=3.20.3,<5" "termcolor>=2.4.0" "wrapt==1.14.1" "grpcio>=1.59.0,<2" \
    "tensorboard>=2.15,<2.16" "tensorflow-estimator>=2.15,<2.16" "keras>=2.15,<2.16" \
    "tensorflow-io-gcs-filesystem>=0.34.0" "overrides>=7.7.0" "easydict>=1.10" && \
    uv pip install --python /.venv/bin/python "matplotlib==3.7.0" "mediapy>=1.2.0" "wandb"

# openpi is mounted at /app, ELASTIC at /elastic, V-GPS at /vgps (see docker/pi05/compose.yml).
ENV PYTHONPATH=/app/src:/app/packages/openpi-client/src:/elastic
ENV VGPS_ROOT=/vgps
ENV TFHUB_CACHE_DIR=/openpi_assets/tfhub
CMD /bin/bash -c "/.venv/bin/python -m elastic.pi05.serving.serve $SERVER_ARGS"
