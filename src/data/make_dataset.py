# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Dict

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder


from transformers import AutoFeatureExtractor

from PIL import Image


class BirdsDataset:
    def __init__(
        self,
        input_filepath: str,
        output_filepath: str,
        data_type: str,
        pretrained_name: str,
    ) -> None:
        super(BirdsDataset, self).__init__()

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.data_type = data_type
        # If the processed folder is populated, read the data from there and return!
        # Otherwise...:
        if Path(output_filepath).is_dir():
            try:
                self.data = self.load_ds()
                return
            except FileNotFoundError as e:
                print(e)

        # Creates an class, with members: PIL images, targets and classes (names of folders)
        ds = ImageFolder(input_filepath + "/" + data_type)

        # Create dictionaries that match bird name to target (an int) and vice versa
        self.label2id = {}
        self.id2label = {}
        for i, class_name in enumerate(ds.classes):
            self.label2id[class_name] = str(i)
            self.id2label[str(i)] = class_name

        # I need my data as a dictionary for huggingface Trainer
        self.data = []
        for i in range(len(ds.targets)):
            self.data.append(
                {"image": Image.open(ds.imgs[i][0]), "label": ds.targets[i]}
            )

        # I do not need ds anymore
        del ds

        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)

        normalize = Normalize(
            mean=feature_extractor.image_mean, std=feature_extractor.image_std
        )

        transforms_ = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )

        for data_point in self.data:
            data_point["pixel_values"] = transforms_(data_point["image"].convert("RGB"))
            data_point["label"] = torch.tensor(data_point["label"])
            # Free up some RAM
            del data_point["image"]

    def save_ds(self) -> None:

        # Create the output_dir if it does not exist
        Path(self.output_filepath).mkdir(exist_ok=True)

        imgs_tsnr = []
        labels_tsnr = []
        for data_point in self.data:
            imgs_tsnr.append(data_point["pixel_values"])
            labels_tsnr.append(data_point["label"])

        labels_tsnr = torch.stack(labels_tsnr, 0)
        imgs_tsnr = torch.stack(imgs_tsnr, 0)
        torch.save(
            [imgs_tsnr, labels_tsnr],
            self.output_filepath + "/" + self.data_type + "_processed.pt",
        )

    def load_ds(self) -> List[Dict]:
        data = torch.load(self.output_filepath + "/" + self.data_type + "_processed.pt")
        ds = []
        for i in range(len(data[1])):
            ds.append({"pixel_values": data[0][i], "label": data[1][i]})
        return ds


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

    pretrained_name = "google/vit-base-patch16-224-in21k"

    # train_ds = BirdsDataset(input_filepath, "train", pretrained_name)
    valid_ds = BirdsDataset(input_filepath, output_filepath, "valid", pretrained_name)
    valid_ds.save_ds()
    # test_ds = BirdsDataset(input_filepath, "test")

    return

    train_ds.save_ds(output_filepath)

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
