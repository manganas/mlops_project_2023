from transformers import AutoModelForImageClassification


class MyClassifier:
    def __init__(
        self,
        pretrained_model: str,
        num_labels: int,
        feature_extractor_cache: str,
        **kwargs,
    ) -> None:

        self._model_ = AutoModelForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            cache_dir=feature_extractor_cache,
            torchscript=True,
            **kwargs,
        )

    def get_model(self) -> AutoModelForImageClassification:
        return self._model_
