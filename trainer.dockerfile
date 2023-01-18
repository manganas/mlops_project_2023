# Base image
FROM python:3.9-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \
    apt install git

RUN pip install wandb
# Copy application essential parts
COPY requirements.txt requirements.txt
COPY data.dvc data.dvc
COPY .dvc .dvc
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# Set working directory
WORKDIR /

# Run commands
RUN pip install -r requirements.txt --no-cache-dir

RUN git init
RUN dvc pull

ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]