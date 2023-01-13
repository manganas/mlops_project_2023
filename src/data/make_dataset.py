# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Dict
import pickle

import hydra
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from hydra.core.config_store import ConfigStore
from torch.utils.data import Dataset


from transformers import AutoFeatureExtractor

from src.config import BirdsConfig

cs = ConfigStore.instance()
cs.store("birds_config", node=BirdsConfig)


class BirdsDataset:
    def __init__(
        self,
        input_filepath: str,
        output_filepath: str,
        data_type: str,
        feature_extractor: AutoFeatureExtractor,
    ) -> None:

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.data_type = data_type
        self.feature_extractor = feature_extractor

        # Create the output save folder if it does not yet exist. Optional now with load_dataset
        Path(output_filepath).mkdir(exist_ok=True, parents=True)

        self.__data__ = load_dataset(
            "imagefolder",
            data_dir=input_filepath + f"/{data_type}",
            cache_dir=output_filepath + f"/{data_type}",
        )

        self.num_classes = len(self.__data__["train"].features["label"].names)

        labels = self.__data__["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label

        if data_type.lower().strip() == "train":
            self.save_labels_ids(label2id, "label2id")
            self.save_labels_ids(id2label, "id2label")

    def get_data(self) -> Dataset:
        # 'train' key is the default for load_dataset. All of the dataset types have it
        return self.__data__["train"]

    def save_labels_ids(self, labels_ids: Dict, filename: str) -> None:
        with open(self.output_filepath + "/" + filename + ".pkl", "wb") as f:
            pickle.dump(labels_ids, f)


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: BirdsConfig) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_filepath = cfg.dirs.input_path
    output_filepath = cfg.dirs.output_path
    pretrained_feature_extractor = cfg.hyperparameters.pretrained_feature_extractor
    feature_extractor_cache_dir = cfg.dirs.feature_extractor

    # Because it is needed in the dataset contrsuctor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_feature_extractor, cache_dir=feature_extractor_cache_dir
    )

    # Prepare train dataset
    _ = BirdsDataset(input_filepath, output_filepath, "train", feature_extractor)

    # Prepare validation dataset
    _ = BirdsDataset(input_filepath, output_filepath, "valid", feature_extractor)

    # Prepare test dataset
    _ = BirdsDataset(input_filepath, output_filepath, "test", feature_extractor)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
