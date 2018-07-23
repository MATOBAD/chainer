FROM python:3.6.6-jessie

COPY . /cnn

WORKDIR /cnn

RUN pip install --upgrade pip
RUN pip install chainer
