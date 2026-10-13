FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install OS dependencies if required for OpenCV/image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY classification_service/ ./classification_service/
COPY app/ ./app/
COPY models/ ./models/
COPY .env .env

EXPOSE 8001

CMD ["uvicorn", "classification_service.main:app", "--host", "0.0.0.0", "--port", "8001"]