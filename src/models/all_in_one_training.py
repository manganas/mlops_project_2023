import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from pathlib import Path

from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import List


# Part of hydra config!
vit_pretrained = "google/vit-base-patch16-224-in21k"
GPU = False
num_workers = 4

# Device defn
device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")

# Prepare dataset
train_data_path = Path.cwd() / "data" / "raw" / "train"
valid_data_path = Path.cwd() / "data" / "raw" / "valid"

means = (0.0) * 3
stds = (1.0) * 3
transform_ = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(means, stds)]
)

train_ds = ImageFolder(train_data_path, transform=transform_)
valid_ds = ImageFolder(train_data_path, transform=transform_)

# Create the dataloaders (probably left in the training file)
feature_extractor = ViTFeatureExtractor().from_pretrained(
    vit_pretrained
)  # Should be the same as for the ViT model!


# Model definition


class ViTClassifier(nn.Module):
    def __init__(
        self, hidden_layers: List[int], n_classes: int, dropout: float = None
    ) -> None:
        super(ViTClassifier, self).__init__()

        if not isinstance(hidden_layers, List):
            hidden_layers = [hidden_layers]

        self.n_classes = n_classes
        self.vit = ViTModel.from_pretrained(vit_pretrained)

        self.Fc = nn.Identity()
        if len(hidden_layers) > 1:
            hidden_layers_list = []
            for layer_in, layer_out in zip(hidden_layers[:-1], hidden_layers[1:]):
                hidden_layers_list.append(nn.Linear(layer_in, layer_out))
                if dropout:
                    hidden_layers_list.append(nn.Dropout(dropout))

            self.Fc = nn.Sequential([*hidden_layers_list])

        self.output_layer = nn.Linear(hidden_layers[-1], n_classes)

    def forward(self, x):
        x = self.vit(pixel_values=x)
        x = self.Fc(x)
        x = self.output_layer(x)

        return x
