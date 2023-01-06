# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from typing import Union, Callable, Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import csv

from PIL import Image

import matplotlib.pyplot as plt


class BirdsDataset(Dataset):
    """
    Class that preprocesses the Birds dataset from Kaggle:

    """

    def __init__(self, in_folder: str, out_folder: str, data_type: str) -> None:
        super(BirdsDataset, self).__init__()

        self.data_type = data_type

        self.in_folder = in_folder
        self.out_folder = out_folder

        if out_folder:
            try:
                self.load_preprocessed()
                print("Loaded preprocessed data")
                return
            except:  # FileNotFoundError probably
                print("No preprocesed data found. Generating...")
                pass

        self.image_paths, self.targets = self.read_raw_data()

        means = (0.0, 0.0, 0.0)
        stds = (1.0, 1.0, 1.0)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(means, stds)]
        )

        self.create_tensors()

    def load_preprocessed(self) -> None:
        self.images, self.targets = torch.load(
            self.out_folder + "/" + self.data_type + "_processed.pt"
        )
        return

    def create_tensors(self):
        content = []
        for image_path in self.image_paths:

            # Read image using PIL Image
            img = Image.open(self.in_folder + "/" + image_path)

            # Pass the PIL image through the transformation
            img = self.transform(img)
            content.append(img.view(1, *img.shape))

            # Save it in a tensor alongside the corresponding target

        imgs_tensor = torch.cat(content, 0)
        print(f"{len(content)} images")
        print(f"Images tensor shape: {imgs_tensor.shape}")

    def read_raw_data(self):
        try:
            with open(f"{self.in_folder}/birds.csv", "r") as f:
                all_birds_csv = csv.reader(f)

                image_paths = []
                targets = []

                for line_ in all_birds_csv:
                    if line_[-1] == self.data_type:
                        image_paths.append(line_[1])

                        # Use Linean scientific nomeclature for targets. Use line_[2] for common names (in capital)
                        targets.append(line_[3])

                return image_paths, targets

        except FileExistsError as e:
            print(e)
            print('Run "make data" again')
            return None, None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self) -> int:
        pass


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # train = BirdsDataset(input_filepath, output_filepath, "train")
    # test = BirdsDataset(input_filepath, output_filepath, "test")
    validation = BirdsDataset(input_filepath, output_filepath, "valid")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
