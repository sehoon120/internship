from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests
import torch
from PIL import Image
from torchvision import transforms

# 기본 전처리 정의 (ImageNet 기준)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# 모델과 전처리기 로드
model_name = "hustvl/Vim-tiny"
# extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)

# 이미지 불러오기
url = 'https://ko.wikipedia.org/wiki/%EB%A7%98%EB%B0%94#/media/%ED%8C%8C%EC%9D%BC:Mamba_Dendroaspis_angusticeps.jpg'
# "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification/kitten.webp"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image = Image.open("your_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

# 전처리 및 예측
# inputs = extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(pixel_values=input_tensor)
logits = outputs.logits
pred = logits.softmax(dim=-1).argmax(dim=-1).item()
print("예측 클래스 ID:", pred)
