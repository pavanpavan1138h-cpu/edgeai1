FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

COPY requirements-runtime.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser model ./model

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "app.py"]
