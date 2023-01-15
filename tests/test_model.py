from tests import _PROJECT_ROOT


from transformers import AutoModelForImageClassification
from src.models.model import MyClassifier


def test_classifier():
    pretrained_name = "google/vit-base-patch16-224-in21k"
    cache_dir = _PROJECT_ROOT + "/models/feature_exractor"
    number_of_classes = 10

    model = MyClassifier(
        pretrained_name, number_of_classes, feature_extractor_cache=cache_dir
    ).get_model()

    assert False, f",{type(model)}"

    assert isinstance(
        model, AutoModelForImageClassification
    ), f"Model is not of class AutoModelForImageClassification"
