import os
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

app = Flask(__name__)

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
num_feats = model.classifier[2].in_features
model.classifier[2] = torch.nn.Linear(num_feats, len(class_names)) 

model.load_state_dict(torch.load("pokemon_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = ConvNeXt_Base_Weights.DEFAULT.transforms()

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return
    return render_template("index.html")