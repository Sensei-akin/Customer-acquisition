FROM tensorflow/tensorflow:2.0.0-py3
MAINTAINER Akinwande Komolafe<akomolafe@terragonltd.com>

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl


COPY requirements.txt /opt/program/requirements.txt

RUN pip install -r /opt/program/requirements.txt


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
# ENV MODEL_PATH="/opt/ml/model"
# ENV EPOCHS=10
# ENV LEARNING_RATE=0.01
# ENV BATCH_SIZE=128
# ENV GPU_COUNT=0


# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /container /opt/program
WORKDIR /opt/program

RUN chmod +x /opt/program/train /opt/program/serve

