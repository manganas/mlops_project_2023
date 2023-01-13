from typing import Dict


from transformers import AutoModelForImageClassification


class MyClassifier:
    def __init__(
        self,
        pretrained_model: str,
        num_labels: int,
        feature_extractor_cache: str,
        **kwargs,
    ) -> None:

        # id2label = {id: label for (label, id) in label2id.items()}

        self._model_ = AutoModelForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            cache_dir=feature_extractor_cache,
            **kwargs,
        )

    def get_model(self) -> AutoModelForImageClassification:
        return self._model_
