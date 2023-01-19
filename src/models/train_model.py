import logging
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_scheduler

from src.data.make_dataset import BirdsDataset
from src.models.model import MyClassifier


def prepare_dataloader(dataset, feature_extractor, kwargs):
    def tokenize_function(examples) -> Dict:
        return feature_extractor(examples["image"])

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["image"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return DataLoader(dataset, **kwargs)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):

    #############
    ## GLOBALS ##

    # Directories
    print(f"configuration Hparams: \n {cfg.experiment.hyperparameters}")
    print(f"configuration Hoptimizer: \n {cfg.optimizers.Optimizer}")

    hparams = cfg.experiment.hyperparameters
    names = cfg.experiment.names
    hoptimizer = cfg.optimizers.Optimizer
    hdirs = cfg.experiment.dirs

    data_input_filepath = hdirs.input_path
    data_output_filepath = hdirs.output_path
    feature_extractor_cache = hdirs.feature_extractor

    saved_models_dir = hdirs.saved_models_dir
    Path(saved_models_dir).mkdir(exist_ok=True, parents=True)
    saved_weights_dir = hdirs.saved_weights_dir

    # Hyperparameters
    print()
    pretrained_model = hparams.pretrained_feature_extractor
    lr = hparams.lr
    batch_size = hparams.batch_size
    epochs = hparams.epochs
    gpu = hparams.gpu
    save_per_epochs = hparams.save_per_epochs
    seed = hparams.seed
    n_train_datapoints = hparams.n_train_datapoints
    n_valid_datapoints = hparams.n_valid_datapoints

    saved_model_name_prefix = names.saved_model_name_prefix

    torch.manual_seed(hparams.seed)
    #############
    #############

    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    device = torch.device(device)

    # Init wandb
    wandb.init(entity="team31", project="full_training")

    # In the dataset class!
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model, cache_dir=feature_extractor_cache
    )

    # Prepare datasets
    train_data = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="train",
    )

    train_dataset = train_data.get_data()

    valid_data = BirdsDataset(
        input_filepath=data_input_filepath,
        output_filepath=data_output_filepath,
        data_type="valid",
    )

    valid_dataset = valid_data.get_data()

    # Prepare data_loaders
    train_loader_options = {"shuffle": True, "batch_size": batch_size, "num_workers": 4}

    train_dataloader = prepare_dataloader(
        train_dataset, feature_extractor, train_loader_options
    )

    valid_loader_options = {
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": 4,
    }

    valid_dataloader = prepare_dataloader(
        valid_dataset, feature_extractor, valid_loader_options
    )

    model_options = {"ignore_mismatched_sizes": True}

    model = MyClassifier(
        pretrained_model=pretrained_model,
        num_labels=train_data.num_classes,
        feature_extractor_cache=feature_extractor_cache,
        **model_options,
    ).get_model()

    wandb.watch(model, log_freq=100)

    # model_name = pretrained_model.split("/")[-1]

    optimizer_dict = {
        "AdamW": torch.optim.AdamW(model.parameters(), lr=lr),
        "SGD": torch.optim.SGD(model.parameters(), lr=lr),
    }

    optimizer = optimizer_dict[hoptimizer[0].optimizer]

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
            # torch.save(
            #     model, saved_models_dir + f"/{saved_model_name_prefix}-{epoch}.pt"
            # )
            torch.save(model, saved_models_dir + f"/{saved_model_name_prefix}.pt")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("training model")
    main()
