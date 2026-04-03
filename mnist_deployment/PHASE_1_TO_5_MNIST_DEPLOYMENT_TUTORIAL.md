# 🚀 MNIST Handwriting Classifier: From Notebook Training to Live Cloud Deployment

This document presents a polished end-to-end walkthrough of how an MNIST deep learning model was trained in Jupyter, packaged into a web application, and deployed publicly on Google Cloud Run.

## 🌐 Live Demo

**Try the app here:**  
https://mnist-app-521506040142.us-central1.run.app/

---

## 📌 Project Summary

This project started with model experimentation in `notebooks/cnn_experiments.ipynb`, where several PyTorch neural networks were trained and compared on the MNIST handwritten digit dataset.

After improving the notebook to save the trained model as a `.pth` checkpoint, the workflow continued through five deployment phases:

1. **Project setup**
2. **Backend API development with FastAPI**
3. **Frontend drawing interface**
4. **Docker containerization**
5. **Deployment to Google Cloud Run**

The result is a live interactive app where users can draw digits in the browser and see the model predict them in real time.

---

## 🧠 Phase 0: Train and Save the Model in Jupyter

The machine learning experimentation took place in:

- `notebooks/cnn_experiments.ipynb`

Multiple architectures were tested, including:

- `BasicCNN`
- `RegularizedCNN`
- `AugmentedCNN`
- `ScheduledCNN`
- `AdvancedCNN`

These experiments helped compare performance improvements across different deep learning strategies such as:

- convolutional feature extraction
- batch normalization and dropout
- data augmentation
- learning rate scheduling
- residual connections

### Saving the trained model

To prepare the model for deployment, the notebook was updated to save the trained weights.

```python
cnn4_accuracy = evaluate_model(cnn4_model, testloader)
print(f'Scheduled CNN Accuracy: {cnn4_accuracy:.2f}%\n')

cnn4_model.to('cpu')
torch.save(cnn4_model.state_dict(), 'mnist_cnn_weights.pth')
print("Model weights saved successfully!")
```

For the final deployed app, the saved checkpoint used was:

- `mnist_advanced_cnn_weights.pth`

---

## 🗂️ Phase 1: Project Setup

A dedicated deployment folder was created to keep the application code separate from the notebook experiments.

### Deployment structure

```text
mnist_deployment/
│
├── static/
│   └── index.html
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── mnist_advanced_cnn_weights.pth
```

### Why this matters

This structure makes the project easier to manage and deploy:

- `main.py` contains the backend inference logic
- `static/index.html` provides the browser-based UI
- `requirements.txt` tracks Python dependencies
- `Dockerfile` defines the runtime container
- the `.pth` file stores the trained model weights

---

## ⚙️ Phase 2: Backend API with FastAPI

The backend was built in:

- `mnist_deployment/main.py`

### Backend responsibilities

The API is responsible for:

- loading the trained model weights
- reconstructing the neural network architecture
- preprocessing user drawings into MNIST-compatible format
- performing inference with PyTorch
- returning predictions through HTTP endpoints

### Key endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/` | `GET` | Serves the web interface |
| `/predict` | `POST` | Accepts an image and returns the digit prediction |
| `/health` | `GET` | Confirms the service is running |

### Example health response

```json
{
  "status": "ok",
  "model": "AdvancedCNN",
  "weights_file": "mnist_advanced_cnn_weights.pth",
  "device": "cpu"
}
```

---

## 🎨 Phase 3: Frontend Drawing Interface

The frontend was implemented in:

- `mnist_deployment/static/index.html`

### Features included

- an interactive drawing canvas
- **Predict** and **Clear** buttons
- live display of:
  - predicted digit
  - confidence score
  - model information
- support for both mouse and touch input

### Why this phase is important

This is where the model becomes visible in action. Instead of only reviewing notebook metrics, users can interact with the model directly by drawing digits and observing how it responds.

---

## 🐳 Phase 4: Docker Containerization

The application was containerized using:

- `mnist_deployment/Dockerfile`
- `mnist_deployment/.dockerignore`

### Benefits of containerization

Docker makes the app:

- reproducible across machines
- easier to test locally
- ready for cloud deployment

### Local run example

```bash
cd mnist_deployment
docker build -t mnist-app-local .
docker run --rm -p 8080:8080 mnist-app-local
```

Then open:

```text
http://localhost:8080
```

This step confirms the full application works before deploying to the cloud.

---

## ☁️ Phase 5: Deploy to Google Cloud Run

The final phase was deploying the app to **Google Cloud Run**, a serverless platform for running containerized applications.

### Deployment configuration

- **Project ID:** `automatic-ace-488412-a7`
- **Region:** `us-central1`
- **Service name:** `mnist-app`

### Deployment command

```bash
gcloud run deploy mnist-app \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080
```

### Final production URL

```text
https://mnist-app-521506040142.us-central1.run.app/
```

### Verification

The deployment was checked with:

```bash
curl "$(gcloud run services describe mnist-app --region us-central1 --format='value(status.url)')/health"
```

This confirmed the service was live and serving the model correctly.

---

## 👀 Visualizing the Model at Work

To see the deployed model in action:

1. open the live app:  
   **https://mnist-app-521506040142.us-central1.run.app/**
2. draw a handwritten digit on the canvas
3. click **Predict**
4. observe the model’s prediction and confidence score

This creates a simple but effective real-time visualization of a deep learning model performing inference on user-generated input.

---

## 🏗️ Architecture at a Glance

```text
Jupyter Notebook Training
        ↓
Saved PyTorch Weights (.pth)
        ↓
FastAPI Inference Backend
        ↓
HTML/CSS/JS Frontend Canvas
        ↓
Docker Container
        ↓
Google Cloud Run Deployment
```

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **FastAPI**
- **HTML / CSS / JavaScript**
- **Docker**
- **Google Cloud Run**
- **Jupyter Notebook**

---

## 🎯 Learning Outcomes

This project demonstrates practical experience in:

- deep learning experimentation and evaluation
- model serialization for deployment
- backend API development for ML inference
- frontend interaction design for model visualization
- containerization using Docker
- cloud deployment using Google Cloud Run

---

## ✅ Conclusion

This end-to-end workflow shows how to move from:

1. training and comparing models in Jupyter
2. saving the best model checkpoint
3. serving the model through an API
4. building an interactive browser interface
5. deploying the final app to the cloud

The outcome is a complete and shareable machine learning application that allows anyone to visually test the trained model in real time.

### Live Demo

**https://mnist-app-521506040142.us-central1.run.app/**
