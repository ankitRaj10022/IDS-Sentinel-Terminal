FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt requirements-dnn.txt requirements-classical.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-app.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "ids_app.api:app", "--host", "0.0.0.0", "--port", "8000"]

