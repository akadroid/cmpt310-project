import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("results/muon_best_model.pt", map_location=device)
class_names = checkpoint["class_names"]

model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
num_feats = model.classifier[2].in_features
model.classifier[2] = torch.nn.Sequential(
    torch.nn.Dropout(0.5),         
    torch.nn.Linear(num_feats, len(class_names))  
)

model.load_state_dict(checkpoint["model_state_dict"])
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

@app.route("/", methods=["POST"])
def predict():
    
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = prediction(filepath)
    print(result)
    return jsonify({"prediction": result})

@app.route("/", methods=["GET"])
def index():
    return render_template("pokdedex.html")

if __name__ == "__main__":
    app.run(debug=True)

    