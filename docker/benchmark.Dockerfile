FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 corral-terminal \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /workspace corral-terminal

WORKDIR /opt/corral
COPY . /opt/corral
RUN python -m pip install --no-cache-dir -e .

VOLUME ["/workspace"]
CMD ["corral", "--help"]
