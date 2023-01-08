# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BirdsDataset:
    def __init__(self, input_filepath: str, data_type: str) -> None:

        self.input_filepath = input_filepath
        self.data_type = data_type

        self.ds = ImageFolder(input_filepath + "/" + data_type)

        self.label2id = {}
        self.id2label = {}

        for i, class_name in enumerate(self.ds.classes):
            self.label2id[class_name] = str(i)
            self.id2label[str(i)] = class_name

        means = (0.0) * 3
        stds = (1.0) * 3

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(means, stds)]
        )

    def save_ds(self, output_filepath: str) -> None:
        imgs_tsnr = []
        labels_tsnr = []
        for data_point in self.ds:
            imgs_tsnr.append(self.transform(data_point[0]))
            labels_tsnr.append(data_point[1])

        labels_tsnr = torch.from_numpy(np.array(labels_tsnr))
        imgs_tsnr = torch.stack(imgs_tsnr, 0)
        torch.save(
            [imgs_tsnr, labels_tsnr],
            output_filepath + "/" + self.data_type + "_processed.pt",
        )


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors="pt")
        encodings["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    Path(output_filepath).mkdir(exist_ok=True)

    train_ds = BirdsDataset(input_filepath, "train")
    train_ds.save_ds(output_filepath)

    valid_ds = BirdsDataset(input_filepath, "valid")
    valid_ds.save_ds(output_filepath)

    test_ds = BirdsDataset(input_filepath, "test")
    test_ds.save_ds(output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
