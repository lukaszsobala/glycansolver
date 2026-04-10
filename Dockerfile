FROM python:3.14-slim

WORKDIR /app

ENV GLYCANSOLVER_USAGE_DB=/app/usage.db

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

EXPOSE 5000

CMD ["gunicorn", "--workers", "1", "-b", "0.0.0.0:5000", "glycansolver.web:app"]
