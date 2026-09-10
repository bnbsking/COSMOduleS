FROM python:3.12-slim

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        procps \
        vim \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        build-essential

WORKDIR /app
COPY . /app

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install --no-root --no-interaction --no-ansi
