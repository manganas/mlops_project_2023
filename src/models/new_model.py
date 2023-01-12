from transformers import ViTModel, ViTFeatureExtractor, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from PIL import Image
from vine import transform

from pathlib import Path

from tqdm import tqdm

vit_pretrained = "google/vit-base-patch16-224-in21k"
in_dir = Path.cwd() / 'data'/'raw'/'train'
batch_size = 64
epochs = 1
lr = 1e-3
model = ViTModel.from_pretrained(vit_pretrained)
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_pretrained)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class BirdsDataset(Dataset):
    def __init__(self, in_dir, feature_extractor, h=None,w=None):
        super(BirdsDataset, self).__init__()
        self.in_dir = in_dir
        
        means = (0.0)*3
        stds = (1.0)*3
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

        self.data = ImageFolder(in_dir)

        self.classes = self.data.classes

        self.h = h
        self.w = w
        self.feature_extractor = feature_extractor
        

    def __getitem__(self, idx):
        data = self.data.imgs[idx]
        img_path, target = data[0], data[1]
        img = Image.open(img_path)
        
        if self.h and self.w:
            img = img.resize(size=(self.h, self.w))

        # img = self.transform(img) ## In favour of feature extractor. The result is the same
        target = torch.tensor(target)
        return  self.feature_extractor(images=[img], return_tensors="pt").pixel_values, target

    def __len__(self):
        return len(self.data)




class MyClassifier(nn.Module):
    def __init__(self, num_classes, feature_extractor):
        super(MyClassifier, self).__init__()
        self.transformer = ViTModel.from_pretrained(vit_pretrained)

        transformer_out_shape = self.get_transformer_embedding_size(torch.rand([1,3,feature_extractor.size, feature_extractor.size]))

        self.output_layer = nn.Linear(transformer_out_shape, num_classes)


    def forward(self, x):
        pass

    def get_transformer_embedding_size(self, x):
        print(type(self.transformer(x)))
        return None



feature_extractor = ViTFeatureExtractor(vit_pretrained)
data = BirdsDataset(in_dir.as_posix(),feature_extractor)
trainloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

model = MyClassifier(len(data.classes),feature_extractor)

# criterion = nn.NLLLoss()
# optim = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for imgs, targets in tqdm(trainloader, desc='Training', leave=False):
        print(imgs.shape, targets.shape)

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
