# Use a lightweight official Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 as required by Google Cloud Run
EXPOSE 8080

# Command to run the application using Uvicorn
# The --host 0.0.0.0 part is essential for Cloud Run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]