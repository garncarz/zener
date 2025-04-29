FROM python:slim

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
