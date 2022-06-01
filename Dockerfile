FROM python:3.9.0

WORKDIR /app

COPY . /app

RUN pip install .

EXPOSE $PORT

CMD [ "theia", "server", "start" ]
