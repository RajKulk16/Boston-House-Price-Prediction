FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install - requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --blind 0.0.0.0:$PORT app:app