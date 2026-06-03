# Multi-stage compilation environment to guarantee ultra-light container footprint
FROM python:3.11-slim AS compiler

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=compiler /root/.local /root/.local
COPY main.py .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Production Security Compliance: Run container as non-root
RUN useradd -u 1005 kona_worker && chown -R kona_worker:kona_worker /app
USER kona_worker

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]