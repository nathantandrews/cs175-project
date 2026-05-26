FROM python:3.14-slim

WORKDIR /model

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6006

CMD ["python", "src/main.py"]
