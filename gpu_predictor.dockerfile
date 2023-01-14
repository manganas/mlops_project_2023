# Base image
FROM python:3.9-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install wandb
# Copy application essential parts
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Set working directory
WORKDIR /workspace/

# Run commands
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT [ "python", "-u", "src/models/predict_model.py" ]
