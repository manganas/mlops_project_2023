FROM python:3.9-slim

EXPOSE $PORT
EXPOSE 8000

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements_app.txt /code/requirements_app.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements_app.txt

COPY ./app /code/app
COPY ./extra_files /code/extra_files



# CMD exec uvicorn app.main:app --port 8000 --host 0.0.0.0 --workers 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]