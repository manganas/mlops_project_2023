from tests import _PATH_DATA, _PROJECT_ROOT
import pytest
import os
import torch
from torch.utils.data import DataLoader

from typing import Dict
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np
from src.data.make_dataset import BirdsDataset
from src.models.train_model import prepare_dataloader


from PIL import Image


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA + "/raw"), reason="Data files not found"
)
@pytest.fixture(scope="module")
def dataset():
    valid_dataset_ = BirdsDataset(
        _PATH_DATA + "/raw", _PATH_DATA + "/processed/train", "train"
    )
    return valid_dataset_


@pytest.fixture(scope="module")
def feature_extractor():

    feat_extrac_pretrained_name = "google/vit-base-patch16-224-in21k"
    feat_extr_cache_dir = _PROJECT_ROOT + "/models/feature_extractor"

    feat_extr = AutoFeatureExtractor.from_pretrained(
        feat_extrac_pretrained_name, cache_dir=feat_extr_cache_dir
    )

    return feat_extr


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA + "/raw"), reason="Data files not found"
)
@pytest.mark.parametrize(
    "batch_size, num_workers, shuffle",
    [
        (1, 2, True),
        (32, 4, False),
    ],
)
def test_prepare_dataloader(
    dataset, feature_extractor, batch_size, num_workers, shuffle
):

    dataset_ = dataset.get_data()

    options = {"num_workers": num_workers, "batch_size": batch_size, "shuffle": shuffle}

    data_loader = prepare_dataloader(dataset_, feature_extractor, options)

    assert isinstance(
        data_loader, DataLoader
    ), 'function "prepare_dataloader" does not return a DataLoader object'

    data_point = next(iter(data_loader))
    assert isinstance(data_point, Dict), "Entries in dataloader not of class Dict"
    assert "labels" in list(
        data_point.keys()
    ), '"labels" not in keys of dataloader\'s entries'
    assert "pixel_values" in list(
        data_point.keys()
    ), '"pixel_values" not in keys of dataloader\'s entries'

    assert isinstance(data_point["labels"], torch.Tensor) and isinstance(
        data_point["pixel_values"], torch.Tensor
    ), "Values of dataloader's entries not Tensors"

    assert (
        data_point["labels"].dtype == torch.long
    ), 'Dataloader\'s "labels" not torch.long '

    img = data_point["pixel_values"]

    assert img.shape[0] == batch_size, f"{img.shape}Batch size not correct"
    assert img.dtype == torch.float, "pixel_values not torch.float"

    assert img.shape[1] == 3, "Not RGB image"

    assert (
        img.shape[2] == feature_extractor.size
        and img.shape[3] == feature_extractor.size
    ), "Dataloaders pixel_values dimensions do not match feature extractor"
