FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["streamlit", "run", "app/Audionome.py", "--server.port=8501"]