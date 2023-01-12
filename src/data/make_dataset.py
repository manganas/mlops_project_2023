# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Dict

import click
from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset

from torch.utils.data import Dataset


class BirdsDataset:
    def __init__(
        self,
        input_filepath: str,
        output_filepath: str,
        data_type: str,
    ) -> None:
        super(BirdsDataset, self).__init__()

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.data_type = data_type

        # Create the output save folder if it does not yet exist. Optional now with load_dataset
        Path(output_filepath).mkdir(exist_ok=True, parents=True)

        self.__data__ = load_dataset(
            "imagefolder",
            data_dir=input_filepath + f"/{data_type}",
            cache_dir=output_filepath + f"/{data_type}",
        )

    def get_data(self) -> Dataset:
        return self.__data__["train"]  # 'train' key is the default for load_dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Prepare train dataset
    _ = BirdsDataset(input_filepath, output_filepath, "train")

    # Prepare validation dataset
    _ = BirdsDataset(input_filepath, output_filepath, "valid")

    # Prepare test dataset
    _ = BirdsDataset(input_filepath, output_filepath, "test")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
