FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies (no cache for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Required: set your Hugging Face token at runtime
# docker run -e HF_TOKEN=hf_xxx email_env
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4.1-mini

# Default environment server for OpenEnv (UV-Compliant structure)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
