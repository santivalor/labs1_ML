FROM python:3.9-slim-buster
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]