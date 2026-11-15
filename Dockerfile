FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=3000

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    adduser --disabled-password --gecos "" appuser

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser public ./public
COPY --chown=appuser:appuser fixtures ./fixtures
COPY --chown=appuser:appuser pyproject.toml ./

USER appuser
EXPOSE 3000
CMD ["python", "-m", "src.server"]
