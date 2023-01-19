.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mlops_project_2023
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install test requirements
requirements_test:
	$(PYTHON_INTERPRETER) -m pip install -r requirements_tests.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Train model
train:
	$(PYTHON_INTERPRETER) src/models/train_model.py

## Run predictions on test images
predict_test:
	$(PYTHON_INTERPRETER) src/models/predict_model.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Profile using cProfile and snakeviz for visualization
profile:
	$(PYTHON_INTERPRETER) -m cProfile -o cProfile_file.prof

# Run tests
run_tests: requirements_test
	pytest tests/


## Docker

## Training
## Build training docker image
docker_training_image:
	docker build -f docker/trainer.dockerfile . -t docker_trainer:latest

## Run latest docker training image
docker_run_trainer:
	@echo "Name of docker run instance: "; \
    read NAME; \
	docker run -e WANDB_API_KEY=a009ef7ac8f8292a33c66a257ee94ec14d28d959 --name $$NAME -v $(pwd)/models/:/models/ docker_trainer:latest

## Build training docker image GPU
gpu_docker_training_image:
	docker build -f docker/gpu_trainer.dockerfile . -t gpu_docker_trainer:latest

## Run latest docker training image GPU
gpu_docker_run_trainer:
	@echo "Name of docker run instance: "; \
    read NAME; \
	docker run -e WANDB_API_KEY=b1b5623638ce4f864549651f863460a2c4f1c940 --name $$NAME --gpus all -v $(pwd)/models/:/models/ gpu_docker_trainer:latest

## Prediction
## Build prediction docker image
docker_prediction_image:
        docker build -f docker/predictor.dockerfile . -t docker_predictor:latest

## Run latest docker prediction image
docker_run_predictor:
        @echo "Name of docker run instance: "; \
    read NAME; \
        docker run -e WANDB_API_KEY=a009ef7ac8f8292a33c66a257ee94ec14d28d959 --name $$NAME -v $(pwd)/models/:/models/ docker_predictor:latest

## Build prediction docker image GPU
gpu_docker_prediction_image:
	docker build -f docker/gpu_predictor.dockerfile . -t gpu_docker_predictor:latest

## Run latest docker prediction image GPU
gpu_docker_run_predictor:
	@echo "Name of docker run instance: "; \
	read NAME; \
	docker run -e WANDB_API_KEY=b1b5623638ce4f864549651f863460a2c4f1c940 --name $$NAME --gpus all -v $(pwd)/models/:/models/ gpu_docker_predictor:latest


### Not very usefule, but good to have
uninstall_pre_commit:
	pip install pre-commit \
	&& pre-commit uninstall -t pre-commit -t pre-merge-commit -t pre-push -t prepare-commit-msg -t commit-msg -t post-commit -t post-checkout -t post-merge -t post-rewrite \
	&& pip uninstall pre-commit -y



## Make docker image for local app testing
docker_app_image:
	docker build -f docker/local_app.dockerfile . -t local_app

run_app_container:
	docker run --name docker_app -p 80:80 local_app

docker_build_and_push_app:
	docker build -f docker/google_cloud_app.dockerfile . -t cloud_app
	docker tag cloud_app gcr.io/pelagic-river-374308/cloud_app
	docker push gcr.io/pelagic-river-374308/cloud_app

deploy_app:
	gcloud run deploy classifier-app --image gcr.io/pelagic-river-374308/cloud_app --platform managed --region europe-west1 --allow-unauthenticated

## Covearge
coverage:
	coverage run -m pytest tests/
	coverage report

## Create custom-job
custom_job_cpu:
	gcloud ai custom-jobs create \
	--region=europe-west1 \
	--display-name=test-run-cpu \
	--config=ai_vertex/config_cpu.yaml

custom_job_gpu:
	gcloud ai custom-jobs create \
	--region=europe-west1 \
	--display-name=test-run-gpu \
	--config=ai_vertex/config_gpu.yaml


## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
