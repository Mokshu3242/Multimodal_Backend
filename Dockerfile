FROM python:3.11-slim

WORKDIR /app

RUN apt update -y && apt upgrade -y && apt install ffmpeg -y

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["fastapi","run","fast_api.py","--host=0.0.0.0","--port=8000"]

