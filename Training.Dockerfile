FROM docker.io/library/python:3.11@sha256:3293c1c51267035cc7dbde027740c9b03affb5e8cff6220d30b7c970e39b1406
ENV PYTHONUNBUFFERED=1
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

RUN pip install numpy==2.2.1
RUN pip install pandas
COPY ./modeling /code/modeling
COPY ./data /code/data

WORKDIR /code

