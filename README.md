Bird species image classification using Vision Transformer 
==============================

This repository contains the project work of Team 31 for the DTU course [Machine Learning Operations](https://kurser.dtu.dk/course/02476) for the year 2023.

Team 31 members:
- Charidimos Vradis s212441
- Georgios Panagiotopoulos s223306
- Ioannis Manganas s220493
- Orfeas Athanasiadis Salales s212584

### Overall goal
The goal of the project is to fine tune a deep learning model based on [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)
 that classifies bird species.
 
### Framework
We plan to use the tranformer framework from [Huggingface](https://huggingface.co/). Specifically, use the Vision Transformer based on the paper:

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

### How to include the framework
We want to use the Transformers framework that includes many pretrained models, which we intend to use in order to transfer learn and train on our dataset.

### Dataset
We plan to use the [BIRDS 450 SPECIES - IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) dataset from Kaggle. This is a dataset containing 80762 images (76262 train, 2250 test, 2250 validation) of size 224 X 224. The images are of birds within 450 species.


### Deep learning models
We expect to use the [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) model, which is a deep learning model and is a transformer that is targeted at vision processing tasks such as image recognition. 

We might as well also try the [BERT Pre-Training of Image Transformers (BEiT)](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/beit) and/or [Data-efficient Image Transformers (DeiT)](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/deit) models, which are follow-up works on the original ViT model.

### Checklist
See [CHECKLIST.md](https://github.com/manganas/mlops_project_2023/blob/main/CHECKLIST.md)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
