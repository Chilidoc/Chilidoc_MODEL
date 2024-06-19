FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install gunicorn

COPY . .

EXPOSE 8080
ENV PORT 8080

CMD ["gunicorn", "predict:app", "--bind", "0.0.0.0:8080", "--workers", "4"]