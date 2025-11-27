# app/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./apps

ENV PYTHONUNBUFFERED=1

# Streamlit runs on port 8501 by default
EXPOSE 8501

CMD ["streamlit", "run", "apps/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
