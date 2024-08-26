# Dockerfile, Image, Container
#FROM --platform=linux/amd64 pytorch/pytorch
FROM --platform=linux/amd64 python:3.12

ENV PYTHONBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ADD inference_challenge.py /opt/app
ADD resources /opt/app/resources

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user resources/nnUNET /opt/app/resources/nnUNET

ENV nnUNet_raw="/opt/app/resources/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/app/resources/nnUNet_preprocessed"
ENV nnUNet_results="/opt/app/resources/nnUNet_results"

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -e /opt/app/resources/nnUNET/

COPY --chown=user:user inference_challenge.py /opt/app/

CMD ["python", "inference_challenge.py"]