FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to container
COPY . .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
# 8000 for FastAPI
# 8501 for Streamlit
EXPOSE 8000 8501

# Copy and grant execution rights to startup script
RUN chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]
