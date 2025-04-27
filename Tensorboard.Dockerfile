FROM docker.io/library/python:3.11@sha256:3293c1c51267035cc7dbde027740c9b03affb5e8cff6220d30b7c970e39b1406
ENV PYTHONUNBUFFERED=1

RUN pip install tensorboard

