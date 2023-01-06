Bird species image classification using Vision Transformer 
==============================

This repository contains the project work of Team 31 for the DTU course [Machine Learning Operations](https://kurser.dtu.dk/course/02476) for the year 2023.

Team 31 members:
- Charidimos Vradis s212441
- Georgios Panagiotopoulos s223306
- Ioannis Manganas s220493
- Orfeas Athanasiadis Salales s212584

### Overall goal
The goal of the project is to fine a tune a deep learning model based on [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)
 that classifies bird species.
 
### Framework used
We used the tranformer framework from [Huggingface](https://huggingface.co/). Specifically, we used the Vision Transformer based on the paper:

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

### Data used for finetuning
We used the [BIRDS 450 SPECIES - IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) dataset from Kaggle.

### Deep learning models used
We will use transformers pre-trained models and mainly ViT for the preprocess, to process an image into tensor, also the image transformation phase, to make
the model most robust against overfitting, and finally the training.

###Checklist
See [CHECKLIST.md](https://github.com/manganas/mlops_project_2023/blob/main/CHECKLIST.md)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
