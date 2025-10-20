FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend /app/frontend
COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]

