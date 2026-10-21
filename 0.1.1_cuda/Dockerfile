FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 python3-dev python3-pip python3-venv git wget build-essential cmake ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/qwen-tts
RUN . /opt/qwen-tts/bin/activate && \
    /opt/qwen-tts/bin/pip install --no-cache-dir \
        "qwen-tts==0.1.1"

COPY bash.bashrc /etc/bash.bashrc
COPY qwen_tts_* /usr/local/bin/
COPY tts_*.py /opt/qwen-tts-extra/

ENV PYTHONPATH=/opt/qwen-tts-extra
ENV NUMBA_CACHE_DIR=/tmp

ENTRYPOINT ["/bin/bash"]
