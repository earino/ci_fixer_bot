FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ci_fixer_bot

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/

# Switch to non-root user
USER ci_fixer_bot

# Set default command
ENTRYPOINT ["python", "-m", "ci_fixer_bot"]
CMD ["--help"]