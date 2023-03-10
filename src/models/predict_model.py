import logging
import pickle
from pathlib import Path
from typing import List

import hydra
import torch
from PIL import Image
from transformers import AutoFeatureExtractor

from src.models.model import MyClassifier


def get_images_from_paths(image_paths: List[str]):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    return images


class Predictor:
    def __init__(self, cfg):

        # Config file contents
        self.pretrained_extractor_name = (
            cfg.experiment.hyperparameters.pretrained_feature_extractor
        )
        self.feature_extractor_cache = cfg.experiment.dirs.feature_extractor
        self.saved_models_dir = cfg.experiment.dirs.saved_models_dir
        data_output_dir = cfg.experiment.dirs.output_path
        gpu = cfg.experiment.hyperparameters.gpu
        self.saved_prefix = cfg.experiment.names.saved_model_name_prefix

        self.device = torch.device(
            "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        )
        print(f"Using device: {self.device}")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.pretrained_extractor_name, cache_dir=self.feature_extractor_cache
        )

        with open(data_output_dir + f"/id2label.pkl", "rb") as f:
            self.id2label = pickle.load(f)

        self.model = self.get_best_model()
        print(type(self.model))

    def predict(self, image_paths: List[str]) -> List["str"]:
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)
        output_logits = self.model(pixel_values=pixel_values).logits
        preds = torch.argmax(output_logits, dim=-1)  # otherwise dims=1
        preds = preds.detach().cpu().numpy()

        prediction_labels = []
        for el in preds:
            prediction_labels.append(self.id2label[el])
        return prediction_labels

    def get_best_model(self):
        num_saved_mdls = list(
            Path(self.saved_models_dir).glob(f"{self.saved_prefix}-*.pt")
        )
        if (not Path(self.saved_models_dir).is_dir) or (len(num_saved_mdls) == 0):
            logger.info("No saved models found, default loaded")
            return MyClassifier(
                self.pretrained_extractor_name,
                len(self.id2label.keys()),
                self.feature_extractor_cache,
            ).get_model()

        # Find the latest version of the saved models
        paths = []
        versions = []
        for file_ in num_saved_mdls:
            file_ = file_.as_posix()
            paths.append(file_)
            versions.append(int(file_.split("-")[-1].split(".")[0]))

        idx = versions.index(max(versions))
        logger.info(f"Loaded {paths[idx]}")

        return torch.load(paths[idx])


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):

    test_dir = Path(cfg.experiment.dirs.input_path)
    images_to_test = test_dir.glob("test/**/*.jpg")

    predictor = Predictor(cfg)
    preds = predictor.predict(list(images_to_test))

    for el in preds:
        print(el)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("predict")
    main()
