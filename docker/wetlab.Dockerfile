FROM condaforge/miniforge3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY tasks/wetlab/environment.yml /tmp/wetlab-environment.yml
RUN sed -e '/^  - pip:/,$d' -e 's/python>=3.10/python=3.12/' /tmp/wetlab-environment.yml > /tmp/wetlab-conda.yml \
    && conda env create --file /tmp/wetlab-conda.yml \
    && conda clean --all --yes

ENV PATH=/opt/conda/envs/wetlab/bin:$PATH

WORKDIR /opt/corral
COPY . /opt/corral
ARG CORRAL_EXTRAS=""
ARG CORRAL_TASK="wetlab"
RUN sh docker/install-runtime.sh --editable "tasks/${CORRAL_TASK}"

VOLUME ["/workspace"]
CMD ["corral", "--help"]
