from datasets import load_dataset
from datasets import load_metric


from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

import torch
import torchvision.transforms as transforms
from torchvision.transforms import (
    ToTensor,
    Normalize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    Resize,
)

import numpy as np


# FOR CONFIG FILE
# "microsoft/swin-tiny-patch4-window7-224"
model_checkpoint = (
    "google/vit-base-patch16-224-in21k"  # pre-trained model from which to fine-tune
)
batch_size = 32  # batch size for training and evaluation
lr = 5e-5
##############################

train_dataset = load_dataset(
    "imagefolder", data_dir="data/raw/train", cache_dir="data/processed/train"
)["train"]


valid_dataset = load_dataset(
    "imagefolder", data_dir="data/raw/valid", cache_dir="data/processed/valid"
)[
    "train"
]  # train is the default key for load_dataset


labels = train_dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_checkpoint, cache_dir="models/"
)

normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)

train_transforms = transforms.Compose(
    [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)


val_transforms = transforms.Compose(
    [
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


train_dataset.set_transform(preprocess_train)
valid_dataset.set_transform(preprocess_val)


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    cache_dir="models/",
)


model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-eurosat",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


metric = load_metric("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


# train_results = trainer.train()
# # rest is optional but nice to have
# trainer.save_model(output_dir="models/")
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()


# metrics = trainer.evaluate()
# # some nice to haves:
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)


# Inference
from PIL import Image


image = Image.open("data/raw/test/CAPPED HERON/1.jpg")  # from to_test_dataset

pretrained_path = "vit-base-patch16-224-in21k-finetuned-eurosat/checkpoint-9"
feat_ext = AutoFeatureExtractor.from_pretrained(pretrained_path)
model = AutoModelForImageClassification.from_pretrained(pretrained_path)

print(model.config.id2label)

# prepare image for the model
encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# forward pass
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits


predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
