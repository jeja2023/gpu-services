FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_ROOT=/models \
    TZ=Asia/Shanghai

WORKDIR /workspace

RUN sed -i \
        -e 's#http://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
        -e 's#http://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
        /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        tzdata \
        software-properties-common \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
    && python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app /workspace/app
COPY main.py /workspace/main.py
COPY models.yml /workspace/models.yml

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "100"]
