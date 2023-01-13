import logging
from pathlib import Path
from typing import List

from src.models.model import MyClassifier
from src.config import BirdsConfig

from transformers import AutoFeatureExtractor


from torchvision.datasets import ImageFolder

from PIL import Image

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store("birds_config", node=BirdsConfig)


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


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: BirdsConfig):

    ## Globals

    # Dirs
    data_output_filepath = cfg.dirs.output_path
    feature_extractor_cache = cfg.dirs.feature_extractor

    saved_models_dir = cfg.dirs.saved_models_dir
    saved_weights_dir = cfg.dirs.saved_weights_dir

    # Hyperparameters
    pretrained_model = cfg.hyperparameters.pretrained_feature_extractor

    # Others
    saved_prefix = cfg.names.saved_model_name_prefix

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model, cache_dir=feature_extractor_cache
    )

    # get_pretrained(saved_models_dir)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("predict")
    main()
