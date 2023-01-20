# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path
from typing import Dict
import json

import torch
import hydra
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from torchvision.datasets import ImageFolder


class BirdsDataset(Dataset):
    def __init__(
        self,
        input_filepath: str,
        output_filepath: str,
        data_type: str,
        feature_extractor_object: AutoFeatureExtractor,
    ) -> None:
        super(BirdsDataset, self).__init__()
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.data_type = data_type

        self.images = ImageFolder(self.input_filepath + f"/{data_type}")

        self.feature_extractor = feature_extractor_object

        self.num_classes = len(self.images.classes)

        self.label2id = {label: i for i, label in enumerate(self.images.classes)}
        self.id2label = {i: label for i, label in enumerate(self.images.classes)}

    def __getitem__(self, idx: int) -> Dict:
        current = self.images[idx]
        img, label = current[0], current[1]

        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}

    def __len__(self) -> int:
        return len(self.images)

    def save(self) -> None:
        Path(self.output_filepath).mkdir(exist_ok=True, parents=True)

        if self.data_type.lower().strip() == "train":
            id2label = {i: label for i, label in enumerate(self.images.classes)}
            json_file = json.dumps(id2label, indent=2)
            with open(self.output_filepath + "/id2label.pkl", "wb") as f:
                pickle.dump(id2label, f)
            with open(self.output_filepath + "/id2label.json", "w") as f:
                f.write(json_file)


@hydra.main(config_path="../models/conf", config_name="config.yaml")
def main(cfg) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_filepath = cfg.experiment.dirs.input_path
    output_filepath = cfg.experiment.dirs.output_path

    ########

    pretrained_model = cfg.experiment.hyperparameters.pretrained_feature_extractor
    feature_extractor_cache = cfg.experiment.dirs.feature_extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model, cache_dir=feature_extractor_cache
    )
    train_set = BirdsDataset(
        input_filepath, output_filepath, "train", feature_extractor
    )
    train_set.save()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
