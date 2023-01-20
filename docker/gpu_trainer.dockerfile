# Base image
FROM  nvcr.io/nvidia/pytorch:22.12-py3

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install wandb

ENV WANDB_API_KEY=a009ef7ac8f8292a33c66a257ee94ec14d28d959

# Copy application essential parts
COPY requirements.txt requirements.txt
COPY data.dvc data.dvc
COPY setup.py setup.py
COPY src/ src/
COPY dtumlops-374716-d8e76837973a.json dtumlops-374716-d8e76837973a.json

# Set working directory
WORKDIR /workspace/

# Run commands
RUN pip install -r requirements.txt --no-cache-dir
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://birds_bucket/
RUN dvc remote modify myremote url gs://birds_bucket/
RUN export GOOGLE_APPLICATION_CREDENTIALS='dtumlops-374716-d8e76837973a.json'

RUN dvc pull

ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]
