from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import pytest
import os
from src.data.make_dataset import BirdsDataset
from pathlib import Path

mock_dataset = _TEST_ROOT + "/test_dataset"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
@pytest.fixture(scope="module")
def train_dataset():
    train_dataset_ = BirdsDataset(
        mock_dataset + "/raw", mock_dataset + "/processed/train", "train"
    )
    return train_dataset_


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
@pytest.fixture(scope="module")
def valid_dataset():
    valid_dataset_ = BirdsDataset(
        mock_dataset + "/raw", mock_dataset + "/processed/train", "valid"
    )
    return valid_dataset_


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
@pytest.fixture(scope="module")
def test_dataset():
    test_dataset_ = BirdsDataset(
        mock_dataset + "/raw", mock_dataset + "/processed/train", "test"
    )
    return test_dataset_


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
def test_datasets_n_classes(train_dataset, valid_dataset, test_dataset):
    # Test for the total number of classes in each dataset
    assert (
        train_dataset.num_classes >= test_dataset.num_classes
    ), "Test set has more classes than training set"

    assert (
        train_dataset.num_classes >= valid_dataset.num_classes
    ), "Validation set has more classes than training set"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
def test_datasets_classes_inclusion(train_dataset, valid_dataset, test_dataset):
    # Check to see that all the classes in test and valid sets are included in the train set

    train_classes = train_dataset.label2id
    train_classes_list = list(train_classes.keys())
    valid_classes = valid_dataset.label2id
    test_classes = test_dataset.label2id

    for el in valid_classes.keys():
        assert (
            el in train_classes_list
        ), f"{el} class from validation set not in training set"

    for el in test_classes.keys():
        assert el in train_classes_list, f"{el} class from test set not in training set"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/raw"), reason="Data files not found"
)
def test_datasets_output_shapes(train_dataset, valid_dataset, test_dataset):
    train_dataset_ = train_dataset.get_data()
    valid_dataset_ = valid_dataset.get_data()
    test_dataset_ = test_dataset.get_data()

    raw_train_path = Path(mock_dataset) / "raw" / "train"
    raw_valid_path = Path(mock_dataset) / "raw" / "valid"
    raw_test_path = Path(mock_dataset) / "raw" / "test"

    n_training_images = len(list(raw_train_path.glob("**/*.jpg")))
    n_validation_images = len(list(raw_valid_path.glob("**/*.jpg")))
    n_testing_images = len(list(raw_test_path.glob("**/*.jpg")))

    assert (
        n_training_images > n_validation_images
    ), "There are more validation than training images!"
    assert (
        n_training_images > n_testing_images
    ), "There are more testing than training images!"

    assert (
        len(train_dataset_) == n_training_images
    ), "The training dataset class outputs a different number of points than there are images"
    assert (
        len(valid_dataset_) == n_validation_images
    ), "The validation dataset class outputs a different number of points than there are images"
    assert (
        len(test_dataset_) == n_testing_images
    ), "The test dataset class outputs a different number of points than there are images"
