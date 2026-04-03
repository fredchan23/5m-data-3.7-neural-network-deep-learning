import base64
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEVICE = torch.device("cpu")


class PredictionRequest(BaseModel):
    image: str


class ScheduledCNN(nn.Module):
    def __init__(self):
        super(ScheduledCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def resolve_model_and_weights() -> tuple[nn.Module, Path, str]:
    candidates = [
        (BASE_DIR / "mnist_advanced_cnn_weights.pth", AdvancedCNN, "AdvancedCNN"),
        (BASE_DIR / "mnist_cnn_weights.pth", ScheduledCNN, "ScheduledCNN"),
    ]

    for weights_path, model_class, model_name in candidates:
        if weights_path.exists():
            model = model_class().to(DEVICE)
            state_dict = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            return model, weights_path, model_name

    raise FileNotFoundError(
        "No model weights found. Expected `mnist_advanced_cnn_weights.pth` or `mnist_cnn_weights.pth`."
    )


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def preprocess_image(image_data: str) -> torch.Tensor:
    if not image_data:
        raise ValueError("Empty image payload received.")

    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    grayscale_image = alpha_composite.convert("L")
    inverted_image = ImageOps.invert(grayscale_image)

    bbox = inverted_image.getbbox()
    if bbox:
        digit = inverted_image.crop(bbox)
        digit.thumbnail((20, 20))
        canvas = Image.new("L", (28, 28), 0)
        offset = ((28 - digit.width) // 2, (28 - digit.height) // 2)
        canvas.paste(digit, offset)
    else:
        canvas = inverted_image.resize((28, 28))

    return transform(canvas).unsqueeze(0).to(DEVICE)


app = FastAPI(title="MNIST Handwriting Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL, WEIGHTS_PATH, MODEL_NAME = resolve_model_and_weights()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "weights_file": WEIGHTS_PATH.name,
        "device": str(DEVICE),
    }


@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>MNIST backend is running</h1><p>The frontend will be added in the next phase.</p>"


@app.post("/predict")
def predict_digit(payload: PredictionRequest):
    try:
        tensor = preprocess_image(payload.image)
        with torch.no_grad():
            output = MODEL(tensor)
            probabilities = torch.exp(output)
            confidence, prediction = torch.max(probabilities, dim=1)

        return {
            "prediction": int(prediction.item()),
            "confidence": round(float(confidence.item()) * 100, 2),
            "model": MODEL_NAME,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
