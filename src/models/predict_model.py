import logging
from typing import List
from pathlib import Path

from src.models.model import MyClassifier


from transformers import AutoFeatureExtractor
import torch

from PIL import Image

import hydra
import pickle


def get_images_from_paths(image_paths: List[str]):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    return images


# def get_pretrained(saved_models_path: str, version: int = -1) -> str:
#     saved_path = Path(saved_models_path)
#     if not saved_path.is_dir():
#         return None

#     max = 0
#     for folder_ in saved_path.glob(f"*/checkpoint-*"):
#         parts = folder_.parts
#         c_version = int(parts[-1].split("-")[-1])
#         if version > 0 and c_version == version:
#             return "-".join(parts) + f"-{version}"
#         if version > max:
#             max = version

#     return max


class Predictor:
    def __init__(self, cfg):

        # Config file contents
        self.pretrained_extractor_name = (
            cfg.hyperparameters.pretrained_feature_extractor
        )
        self.feature_extractor_cache = cfg.dirs.feature_extractor
        self.saved_models_dir = cfg.dirs.saved_models_dir
        data_output_dir = cfg.dirs.output_path
        gpu = cfg.hyperparameters.gpu
        self.saved_prefix = cfg.names.saved_model_name_prefix

        device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        print(f"Using device: {device}")

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
        output_logits = self.model(pixel_values=pixel_values)
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


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg):

    ### Globals

    # # Dirs
    # data_output_filepath = cfg.dirs.output_path
    # feature_extractor_cache = cfg.dirs.feature_extractor
    # saved_models_dir = cfg.dirs.saved_models_dir

    # # Hyperparameters
    # pretrained_feat_extractor = cfg.hyperparameters.pretrained_feature_extractor

    # # Names
    # saved_model_name_prefix = cfg.names.saved_model_name_prefix

    # # Initiate feature extractor
    # feature_extractor = AutoFeatureExtractor.from_pretrained(
    #     pretrained_feat_extractor, cache_dir=feature_extractor_cache
    # )

    # # Load model: If a checkpoint is found, find the latest version.
    # # Otherwise, instantiate a new from the default feature extractor
    # # and hope for the best

    predictor = Predictor(cfg)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("predict")
    main()
