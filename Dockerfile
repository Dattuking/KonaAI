FROM python:3.11-slim

WORKDIR /app

# Upgrade pip and install dependencies globally in the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend application script
COPY main.py .

# Explicitly grand read/write permissions to the working directory just in case
RUN chmod -R 777 /app

EXPOSE 7860

# Run uvicorn directly via python module routing to bypass folder path blocks
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
