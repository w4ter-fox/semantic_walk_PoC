FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ api/
COPY web/ web/
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
