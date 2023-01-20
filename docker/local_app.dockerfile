FROM python:3.9-slim
WORKDIR /code
COPY ./requirements_app.txt /code/requirements_app.txt

RUN apt-get update

RUN pip install --no-cache-dir --upgrade -r /code/requirements_app.txt
COPY ./app /code/app
COPY ./models /code/models
COPY ./data/processed /code/data/processed

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
