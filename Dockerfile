FROM python:3.11-slim

WORKDIR /app

COPY requirements_server.txt /app/requirements_server.txt
RUN pip install -r requirements_server.txt

COPY fl_platform /app/fl_platform
# COPY kafka-certs /app/kafka-certs
COPY server.py /app/server.py

CMD ["python", "server.py"]
