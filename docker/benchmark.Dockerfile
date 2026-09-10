FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/corral
COPY . /opt/corral
RUN python -m pip install --no-cache-dir -e . \
    && python -m corral.runtime.permissions --prepare-image

VOLUME ["/workspace"]
CMD ["corral", "--help"]
