from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from PIL import Image
from vine import transform

from pathlib import Path

vit_pretrained = "google/vit-base-patch16-224-in21k"
in_dir = Path.cwd() / 'data'/'raw'/'test'


model = VisionEncoderDecoderModel.from_pretrained(vit_pretrained)
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_pretrained)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class BirdsDataset(Dataset):
    def __init__(self, in_dir, h=64,w=64):
        super(BirdsDataset, self).__init__()
        self.in_dir = in_dir
        
        means = (0.0)*3
        stds = (1.0)*3
        self.transform = transforms.Compose([
            transforms.Resize(h,w),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

        self.images = ImageFolder(in_dir)
        print(type(self.images))


    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass




class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()

    def forward(self, x):
        pass



data = BirdsDataset(in_dir.as_posix())


# def predict_step(image_paths):
#     images = []
#     for image_path in image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")

#         images.append(i_image)
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)

#     return
    
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds
