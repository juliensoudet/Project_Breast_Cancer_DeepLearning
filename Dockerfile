#FROM python:3.10.6-buster
FROM tensorflow/tensorflow:2.16.1

WORKDIR /prod
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# ensure all requirements are listed (no tensorflow and keras, as this comes with the env)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY models models

COPY breast_lesion_DL_pack breast_lesion_DL_pack
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
# RUN make reset_local_files


CMD uvicorn breast_lesion_DL_pack.api.fast:app --host 0.0.0.0 --port $PORT
