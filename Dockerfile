FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

USER appuser

# Default: run the API
EXPOSE 8000
CMD ["uvicorn", "whiskerscope.adapters.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
