# End-to-End MLOps: Deploying an MNIST Handwriting App to GCP Cloud Run

This tutorial covers the complete pipeline for taking your trained PyTorch MNIST model (`mnist_cnn_weights.pth`) and deploying it as a full-stack web application on Google Cloud Run.

## Phase 1: Project Setup

Create a new folder on your computer for the deployment project. Your directory structure should look exactly like this:

```text
mnist_deployment/
│
├── static/
│   └── index.html             # The frontend user interface
│
├── main.py                    # The FastAPI backend
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── mnist_cnn_weights.pth      # Your saved PyTorch model weights
```

## Phase 2: The Backend & Preprocessing (FastAPI)

First, define the libraries your container will need.

**1. Create `requirements.txt`:**
```text
fastapi
uvicorn
python-multipart
torch
torchvision
Pillow
```

**2. Create `main.py`:**
This script runs the web server, serves the frontend HTML, and processes incoming image predictions. It requires the `ScheduledCNN` class you used during training so PyTorch knows how to load the weights.

```python
import base64
import io
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# --- 1. DEFINE YOUR MODEL ARCHITECTURE ---
# (Paste your exact ScheduledCNN class definition from your training code here)
class ScheduledCNN(nn.Module):
    def __init__(self):
        super(ScheduledCNN, self).__init__()
        # Example architecture - replace with your exact layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10) # Adjust based on your pooling
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.log_softmax(self.fc1(x))
        return x

# --- 2. INITIALIZE APP & LOAD MODEL ---
app = FastAPI()

# Mount the static directory to serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model weights
device = torch.device('cpu') # Always deploy on CPU unless using specialized GPU cloud instances
model = ScheduledCNN()
model.load_state_dict(torch.load('mnist_cnn_weights.pth', map_location=device))
model.eval()

# --- 3. DEFINE IMAGE PREPROCESSING ---
# The web canvas is black ink on white background.
# MNIST is white ink on black background, 28x28 pixels.
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Use the exact normalization you used during training
])

# --- 4. API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict_digit(request: Request):
    data = await request.json()
    image_data = data['image']
    
    # Decode base64 image from frontend
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    
    # Extract the alpha channel (drawing) and create a white background
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    
    # Convert to grayscale and INVERT colors (make it white on black)
    grayscale_image = alpha_composite.convert("L")
    inverted_image = ImageOps.invert(grayscale_image)
    
    # Transform to tensor
    tensor = transform(inverted_image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(tensor)
        # Assuming model outputs LogSoftmax, get the predicted class
        prediction = output.argmax(dim=1, keepdim=True).item()
        
    return {"prediction": prediction}
```

## Phase 3: The Frontend (HTML5 Canvas)

We need a drawing board that captures the user's mouse/touch events and sends the drawing to the backend.

**3. Create `static/index.html`:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Handwriting Classifier</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; margin-top: 50px; background-color: #f4f4f9; }
        canvas { border: 2px solid #333; background-color: white; border-radius: 8px; cursor: crosshair; }
        .controls { margin-top: 15px; }
        button { padding: 10px 20px; font-size: 16px; margin: 0 5px; cursor: pointer; border: none; border-radius: 4px; background-color: #007bff; color: white; }
        button:hover { background-color: #0056b3; }
        #clear-btn { background-color: #dc3545; }
        #clear-btn:hover { background-color: #c82333; }
        h2 { margin-top: 20px; color: #333; }
    </style>
</head>
<body>

    <h1>Draw a Digit (0-9)</h1>
    <canvas id="canvas" width="280" height="280"></canvas>
    
    <div class="controls">
        <button id="clear-btn">Clear</button>
        <button id="predict-btn">Predict</button>
    </div>

    <h2>Prediction: <span id="result">-</span></h2>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        // Setup canvas context for thick, smooth lines (mimics a thick marker)
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';

        // Drawing events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }

        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').innerText = '-';
        });

        // Predict button
        document.getElementById('predict-btn').addEventListener('click', async () => {
            const dataURL = canvas.toDataURL('image/png');
            
            document.getElementById('result').innerText = 'Thinking...';

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            });

            const data = await response.json();
            document.getElementById('result').innerText = data.prediction;
        });
    </script>
</body>
</html>
```

## Phase 4: Containerization (Docker)

To run this seamlessly on GCP, we package the app into a Docker container.

**4. Create `Dockerfile`:**
```dockerfile
# Use official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
# Install CPU-only PyTorch to keep image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Phase 5: Deployment to GCP Cloud Run

Google Cloud Run is a serverless platform that will host your Docker container. You only pay when someone is actively making a prediction.

**Prerequisites:**
* You must have a Google Cloud Account.
* You must have the [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install) installed on your computer.

**5. Deploy via Terminal:**
Open your terminal, navigate to your `mnist_deployment` folder, and run these commands sequentially:

1.  **Log in to your Google Cloud account:**
    ```bash
    gcloud auth login
    ```

2.  **Set your target Google Cloud Project:**
    *(Replace `YOUR_PROJECT_ID` with your actual GCP project ID)*
    ```bash
    gcloud config set project YOUR_PROJECT_ID
    ```

3.  **Deploy the service using Cloud Run Source Deployment:**
    This command automatically builds the Docker container using Cloud Build and deploys it.
    ```bash
    gcloud run deploy mnist-app --source . --region us-central1 --allow-unauthenticated
    ```

**What happens next?**
* Google will prompt you to confirm the deployment. Press `y` and `Enter`.
* It will take 2-4 minutes to upload the files, build the Docker image, and provision the serverless infrastructure.
* Once finished, the terminal will output a **Service URL** (e.g., `https://mnist-app-xyz123-uc.a.run.app`). 

Click that link, and your handwritten digit classifier is live on the internet!