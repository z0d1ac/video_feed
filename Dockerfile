FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libopenblas-dev \
    liblapack-dev \
    libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5050

EXPOSE 5050


COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["./start.sh"]

