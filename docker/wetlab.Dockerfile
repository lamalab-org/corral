FROM condaforge/miniforge3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 corral-terminal \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /workspace corral-terminal

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY tasks/wetlab/environment.yml /tmp/wetlab-environment.yml
RUN sed '/^  - pip:/,$d' /tmp/wetlab-environment.yml > /tmp/wetlab-conda.yml \
    && conda env create --file /tmp/wetlab-conda.yml \
    && conda clean --all --yes

ENV PATH=/opt/conda/envs/wetlab/bin:$PATH

WORKDIR /opt/corral
COPY . /opt/corral
RUN pip install --no-cache-dir --editable . --editable tasks/wetlab

VOLUME ["/workspace"]
CMD ["corral", "--help"]
