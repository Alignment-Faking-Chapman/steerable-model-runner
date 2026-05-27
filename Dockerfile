FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runner source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

ENTRYPOINT ["python3", "server.py"]
