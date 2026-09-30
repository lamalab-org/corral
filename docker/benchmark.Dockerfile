FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/corral
COPY . /opt/corral
ARG CORRAL_EXTRAS=""
ARG CORRAL_TASK="samplemath"
RUN sh docker/install-runtime.sh --editable "tasks/${CORRAL_TASK}"

VOLUME ["/workspace"]
CMD ["corral", "--help"]
