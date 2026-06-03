FROM python:3.11-slim

WORKDIR /app

# Copy dependency requirements first to leverage caching layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server script directly into the workspace root
COPY main.py .

# Hugging Face strictly requires exposing and binding to port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
