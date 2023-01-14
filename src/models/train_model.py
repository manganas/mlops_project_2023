import logging
from pathlib import Path

from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_metric

from torch.utils.data import DataLoader


from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    get_scheduler,
)


import wandb

from src.data.make_dataset import BirdsDataset
from src.models.model import MyClassifier


def move_to(x, device: torch.device):
    pass


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg):

    #############
    ## GLOBALS ##

    # Directories

    data_input_filepath = cfg.dirs.input_path
    data_output_filepath = cfg.dirs.output_path
    feature_extractor_cache = cfg.dirs.feature_extractor

    saved_models_dir = cfg.dirs.saved_models_dir
    Path(saved_models_dir).mkdir(exist_ok=True, parents=True)
    saved_weights_dir = cfg.dirs.saved_weights_dir

    # Hyperparameters
    pretrained_model = cfg.hyperparameters.pretrained_feature_extractor
    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs
    gpu = cfg.hyperparameters.gpu
    save_per_epochs = cfg.hyperparameters.save_per_epochs
    seed = cfg.hyperparameters.seed
    n_train_datapoints = cfg.hyperparameters.n_train_datapoints
    n_valid_datapoints = cfg.hyperparameters.n_valid_datapoints

    #############
    #############

    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    device = torch.device(device)

    # Init wandb
    wandb.init()

    # In the dataset class!
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model, cache_dir=feature_extractor_cache
    )

    # Prepare datasets
    train_data = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="train",
        feature_extractor=feature_extractor,
    )

    train_dataset = train_data.get_data()

    valid_data = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="valid",
        feature_extractor=feature_extractor,
    )

    valid_dataset = valid_data.get_data()

    # Get features from the data

    def tokenize_function(examples) -> Dict:
        return feature_extractor(examples["image"])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(["image"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")

    valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.remove_columns(["image"])
    valid_dataset = valid_dataset.rename_column("label", "labels")
    valid_dataset.set_format("torch")

    # train_dataset = train_dataset.shuffle(seed=seed).select(range(n_train_datapoints))
    # valid_dataset = valid_dataset.shuffle(seed=seed).select(range(n_valid_datapoints))

    # print(f"Length of training dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)

    model_options = {"ignore_mismatched_sizes": True}

    model = MyClassifier(
        pretrained_model=pretrained_model,
        num_labels=train_data.num_classes,
        feature_extractor_cache=feature_extractor_cache,
        **model_options,
    ).get_model()

    wandb.watch(model, log_freq=100)

    model_name = pretrained_model.split("/")[-1]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Training"):

        running_loss = 0.0
        accuracy = 0.0
        model.train()

        for batch in tqdm(train_dataloader, desc="Batch", leave=False):

            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            y_pred = model(**batch)

            class_pred = torch.argmax(F.softmax(y_pred.logits, dim=1), dim=1)

            is_correct = (
                class_pred.detach().cpu().numpy() == np.array(batch["labels"].cpu())
            ).sum()

            accuracy += is_correct

            loss = y_pred.loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            lr_scheduler.step()

        running_loss /= len(train_dataset)
        accuracy /= len(train_dataset)
        wandb.log({"training loss": running_loss})
        wandb.log({"training accuracy": accuracy})

        print(f"Training Loss: {running_loss}, Training Accuracy: {accuracy}")

        model.eval()
        running_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}

                y_pred = model(**batch)

                class_pred = torch.argmax(F.softmax(y_pred.logits, dim=1), dim=1)

                is_correct = (
                    class_pred.detach().cpu().numpy() == np.array(batch["labels"].cpu())
                ).sum()

                accuracy += is_correct

                loss = y_pred.loss
                running_loss += loss.item()

            running_loss /= len(valid_dataset)
            accuracy /= len(valid_dataset)

            wandb.log({"validation loss": running_loss})
            wandb.log({"validation accuracy": accuracy})
            print(f"Validation Loss: {running_loss}, Validation Accuracy: {accuracy}")

        if epoch % save_per_epochs == 0:
            torch.save(model, saved_models_dir + f"/checkpoint-{epoch}.pt")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("training model")
    main()
