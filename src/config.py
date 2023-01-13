from dataclasses import dataclass


@dataclass
class Directories:
    input_path: str
    output_path: str


@dataclass
class Hyperparameters:
    pretrained_feature_extractor: str
    train_batch_size: int
    num_workers: int
    valid_batch_size: int
    lr: float
    max_epochs: int
    precision: int
    device: str
    num_devices: int
    seed: int


@dataclass
class BirdsConfig:
    dirs: Directories
    hyperparameters: Hyperparameters
