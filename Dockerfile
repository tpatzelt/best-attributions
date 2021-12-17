# set base image (host OS)
FROM python:3.9-slim-buster

RUN apt update && apt install nano git  -y
RUN git clone https://github.com/tpatzelt/best-attributions.git
WORKDIR /best-attributions/

## install dependencies
RUN pip3 install -r requirements.txt

## test run
RUN python run_experiment.py with random_config dataset.num_samples=2
