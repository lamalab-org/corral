FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright

# REBOUND and celerite2 build C extensions from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/corral
COPY . /opt/corral

# Install the locked REBOUND before the task pulls it in, so the release whose
# setup.py misdetects ARM hosts is never the one that gets built.
RUN python tasks/stargazer/scripts/install_rebound.py

ARG CORRAL_EXTRAS=""
ARG CORRAL_TASK="stargazer"
RUN sh docker/install-runtime.sh --editable "tasks/${CORRAL_TASK}"

VOLUME ["/workspace"]
CMD ["corral", "--help"]
