# Use a lightweight official Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip check
RUN find /usr/local/lib/python3.11/site-packages/ -name "flow.py" -ls

# Copy the rest of the application code
COPY api.py .
COPY run.py .
COPY test_anthropic.py .

# Expose port 8080 as required by Google Cloud Run
EXPOSE 8080

# Command to run the application using Uvicorn, respecting the PORT env var from Cloud Run
# The --host 0.0.0.0 part is essential for Cloud Run
CMD uvicorn api:app --host 0.0.0.0 --port $PORT