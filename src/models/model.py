import torch
import torch.nn as nn

from transformers import AutoModelForImageClassification
from typing import Dict


# model = AutoModelForImageClassification.from_pretrained(
#     pretrained_model,
#     label2id=label2id,
#     id2label=id2label,
#     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
#     cache_dir=feature_extractor_cache,
# )


class MyClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        label2id: Dict,
        feature_extractor_cache: str,
        **kwargs
    ) -> None:
        super(MyClassifier, self).__init__()

        id2label = {id: label for (label, id) in label2id.items()}

        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model,
            label2id=label2id,
            id2label=id2label,
            cache_dir=feature_extractor_cache,
            **kwargs
        )

    def forward(self, x: Dict):  # Check the output type!
        return self.model(x)
