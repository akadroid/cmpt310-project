import os
import sys

import numpy as np
import torch
import torchvision
from tqdm.notebook import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
path = 'results/'
best_model_path = os.path.join(path, 'best_model.pt')

test_transforms = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

def get_classes():
    data = torchvision.datasets.ImageFolder(root='data/pokemon')
    class_names = data.classes
    print('classes:', len(class_names))
    return class_names

# https://stackoverflow.com/questions/61001855/how-to-change-dataloader-in-pytorch-to-read-one-image-for-prediction
def safe_pil_loader(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
        except:
            raise ValueError("Path must be valid")
        return res

def create_model():
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    # model.classifer[2] is the linear layer
    num_feats = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(num_feats, len(get_classes()))
    model = model.to('cuda')
    return model

def main():
    path = sys.argv[1]

    image = Image.open(path).convert('RGB')
    image = test_transforms(image).unsqueeze(0).to('cuda')

    model = create_model()
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    # Use model and print predicted category
    with torch.no_grad():
        prediction = model(image).squeeze(0).softmax(0)
        class_names = get_classes()
        
        prob, category_id = torch.topk(prediction, 5)
        
        for i in range(len(category_id)):
            print(f"{i+1}. {class_names[category_id[i]]}: {100 * prob[i].item():.1f}%")

main()

