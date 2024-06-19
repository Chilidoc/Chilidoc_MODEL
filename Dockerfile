FROM python:3.8-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install gunicorn

# Download the model file from HTTP URL
RUN apt-get update && apt-get install -y curl && \
    curl -o best_model.h5 https://storage.googleapis.com/chilidoc-cloud-storage/model/best_model.h5

# Copy the rest of the application code
COPY . .

EXPOSE 8080
ENV PORT 8080

CMD ["gunicorn", "predict:app", "--bind", "0.0.0.0:8080", "--workers", "4"]