# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# docker-compose から引数を受け取る
ARG http_tmp
ARG https_tmp

# プロキシ設定（必要な環境のみ）
ENV http_proxy=${http_tmp}
ENV https_proxy=${https_tmp}

# タイムゾーン
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ===== apt =====
# 学内proxyで security.ubuntu.com が落ちるケースに備え、jammy-security を archive.ubuntu.com 側へ寄せる
RUN sed -i 's|http://security.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
      git curl wget ca-certificates \
      nano tmux htop \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

# ===== Hugging Face cache =====
# compose で /workspace をマウントする想定
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- DGL CUDA 12.1 対応版（必要な場合のみ）----
# PyPI の dgl は CPU 版になることがあるため、公式 wheel repo を利用
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# （任意）GiNZA のモデルをコンテナに同梱したい場合
# RUN python -m spacy download ja_ginza

# ==============================
# ja_ginza_bert_large (URL install)
# ==============================
RUN pip install --no-cache-dir \
    https://github.com/megagonlabs/ginza/releases/download/v5.2.0/ja_ginza_bert_large-5.2.0b1-py3-none-any.whl

# ==============================
# spaCy model check
# ==============================
RUN python -c "import spacy; spacy.load('ja_ginza_bert_large'); print('ja_ginza_bert_large loaded successfully')"
