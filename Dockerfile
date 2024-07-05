# Dockerfile, Image, Container
FROM --platform=linux/amd64 pytorch/pytorch

ENV PYTHONBUFFERED 1

WORKDIR /opt/app

ADD inference.py /opt/app
ADD resources /opt/app/resources

ENV nnUNet_raw="/opt/app/resources/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/app/resources/nnUNet_preprocessed"
ENV nnUNet_results="/opt/app/resources/nnUNet_results"

ADD test /opt/app/test

ADD nnUNET /opt/app/nnUNET
RUN pip install --no-cache-dir -e /opt/app/nnUNET/

CMD ["python", "/opt/app/inference.py"]

