FROM python:3.13.5-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio huggingface_hub

COPY app.py /app/

EXPOSE 7860


CMD ["python", "app.py"]
