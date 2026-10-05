FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verifier search is currently a library, not a server. 
# We'll just run pytest as the default command for the container to prove it works.
CMD ["python", "-m", "pytest", "tests/"]
