FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    ffmpeg \
    curl \
    libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download ONNX models (with fallback mirrors for ArcFace)
RUN mkdir -p /app/models \
    && curl -fsSLo /app/models/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && ( curl -fsSLo /app/models/w600k_r50.onnx \
    https://huggingface.co/vjump21848/buffalo_l_unzip/resolve/main/w600k_r50.onnx \
    || curl -fsSLo /app/models/w600k_r50.onnx \
    https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx \
    || curl -fsSLo /app/models/w600k_r50.onnx \
    https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/w600k_r50.onnx \
    )

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
