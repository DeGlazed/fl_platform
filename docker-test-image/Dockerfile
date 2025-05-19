FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and app code
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]