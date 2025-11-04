# Dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies if any
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     some-lib \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If your main.py is a script that trains and saves, you might just run it
# If it's a web server for inference, you'd use a server like Gunicorn or Uvicorn
# Example for a Flask/FastAPI app serving predictions:
# EXPOSE 8000
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"] # Assuming 'app' is your Flask/FastAPI instance in main.py

# If main.py is just a script to be run on demand:
CMD ["python", "main.py"]
