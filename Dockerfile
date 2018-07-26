FROM python:3.6.6-jessie

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install chainer
