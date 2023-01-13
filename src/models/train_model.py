from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from datasets import load_metric


import torch
from torchvision.transforms import (
    ToTensor,
    Normalize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    Resize,
    Compose,
)

from src.data.make_dataset import BirdsDataset

import hydra
from hydra.core.config_store import ConfigStore
from src.config import BirdsConfig

import numpy as np


cs = ConfigStore.instance()
cs.store("birds_config", node=BirdsConfig)


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: BirdsConfig):

    #############
    ## GLOBALS ##

    # Directories

    data_input_filepath = cfg.dirs.input_path
    data_output_filepath = cfg.dirs.output_path
    feature_extractor_cache = cfg.dirs.feature_extractor

    saved_models_dir = cfg.dirs.saved_models_dir
    saved_weights_dir = cfg.dirs.saved_weights_dir

    # Hyperparameters
    pretrained_model = cfg.hyperparameters.pretrained_feature_extractor
    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs

    #############
    #############

    ## Logic that checks if an already trained model exists!

    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)

    # Prepare datasets
    train_dataset = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="train",
        feature_extractor=feature_extractor,
    ).get_data()

    valid_dataset = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="valid",
        feature_extractor=feature_extractor,
    ).get_data()

    labels = train_dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model, cache_dir=feature_extractor_cache
        )

    normalize = Normalize(
        mean=feature_extractor.image_mean, std=feature_extractor.image_std
    )

    train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
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
        pretrained_model,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        cache_dir=feature_extractor_cache,
    )

    model_name = pretrained_model.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
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

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model(output_dir=saved_models_dir)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics(
        "train",
        train_results.metrics,
    )
    trainer.save_state(output_dir=saved_weights_dir)

    metrics = trainer.evaluate()
    # some nice to haves:
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
