from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification

from src.data.make_dataset import BirdsDataset, ImageClassificationCollator
from src.models.model import Classifier
from src.config import BirdsConfig

import hydra
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()
cs.store("birds_config", node=BirdsConfig)


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: BirdsConfig):
    # Dirs, hyperparameters
    # in_path = Path.cwd() / "data" / "raw"
    # in_path = in_path.as_posix()
    # pretrained_feature_extractor = "google/vit-base-patch16-224-in21k"

    in_path = cfg.dirs.input_path

    hyper = cfg.hyperparameters
    pretrained_feature_extractor = hyper.pretrained_feature_extractor
    train_batch_size = hyper.train_batch_size
    num_workers = hyper.num_workers
    valid_batch_size = hyper.valid_batch_size
    seed = hyper.seed
    lr = hyper.lr
    device = hyper.device
    num_devices = hyper.num_devices
    precision = hyper.precision
    max_epochs = hyper.max_epochs

    # Datasets
    train_dataset = BirdsDataset(in_path, "train")
    valid_dataset = BirdsDataset(in_path, "valid")
    test_dataset = BirdsDataset(in_path, "test")

    train_ds = train_dataset.ds
    valid_ds = valid_dataset.ds
    test_ds = test_dataset.ds

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        pretrained_feature_extractor
    )

    model = ViTForImageClassification.from_pretrained(
        pretrained_feature_extractor,
        num_labels=len(train_dataset.label2id),
        label2id=train_dataset.label2id,
        id2label=train_dataset.id2label,
    )

    collator = ImageClassificationCollator(feature_extractor)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )

    pl.seed_everything(seed)
    classifier = Classifier(model, lr=lr)
    trainer = pl.Trainer(
        accelerator=device,
        devices=num_devices,
        precision=precision,
        max_epochs=max_epochs,
    )
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    main()
