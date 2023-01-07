from transformers import ViTForImageClassification
from src.data.make_dataset import BirdsDataset, ImageClassificationCollator
from src.models.model import Classifier
from transformers import ViTFeatureExtractor, ViTForImageClassification

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pathlib import Path


def main():
    # Dirs, hyperparameters
    in_path = Path.cwd().parent.parent / "data" / "raw"
    print(in_path)
    # in_path = cfg.dirs.input_path

    pretrained_feature_extractor = "google/vit-base-patch16-224-in21k"

    # Datasets
    train_dataset = BirdsDataset(in_path, "train")
    valid_dataset = BirdsDataset(in_path, "valid")
    test_dataset = BirdsDataset(in_path, "test")

    train_ds = BirdsDataset(in_path, "train").ds
    valid_ds = BirdsDataset(in_path, "valid").ds
    test_ds = BirdsDataset(in_path, "test").ds

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        pretrained_feature_extractor
    )

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(train_dataset.label2id),
        label2id=train_dataset.label2id,
        id2label=train_dataset.id2label,
    )

    collator = ImageClassificationCollator(feature_extractor)
    train_loader = DataLoader(
        train_ds, batch_size=8, collate_fn=collator, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(valid_ds, batch_size=8, collate_fn=collator, num_workers=2)

    pl.seed_everything(42)
    classifier = Classifier(model, lr=2e-5)
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision=16, max_epochs=4)
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    print("hi")
    main()
