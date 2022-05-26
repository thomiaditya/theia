FROM python:3

COPY . .

RUN pip install .

CMD [ "theia-recommender", "train" ]
