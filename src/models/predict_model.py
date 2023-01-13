from PIL import Image
import torch

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

image = Image.open("data/raw/test/CAPPED HERON/1.jpg")  # from to_test_dataset

pretrained_path = "vit-base-patch16-224-in21k-finetuned-eurosat/checkpoint-9"
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_path)
model = AutoModelForImageClassification.from_pretrained(pretrained_path)

print(model.config.id2label)

# prepare image for the model
encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# forward pass
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits


predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
