FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY models/ models/
COPY dataset/ dataset/
COPY . .

ENTRYPOINT ["streamlit", "run", "src/Audionome.py", "--server.port=8501"]