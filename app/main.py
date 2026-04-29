from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

import pika
import uuid
import base64
import json

import consul
import socket

# ========== REGISTRATION CONSUL ==========
def register_to_consul(service_name: str, port: int):
    """Enregistre le service auprès de Consul"""
    try:
        c = consul.Consul(host='localhost', port=8500)
        
        # Obtenir l'adresse IP locale
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        # Enregistrer le service
        c.agent.service.register(
            name=service_name,
            service_id=f"{service_name}-{port}",
            address=ip_address,
            port=port,
            check=consul.Check.http(f"http://localhost:{port}/health", interval="10s")
        )
        print(f"✅ Service {service_name} enregistré dans Consul sur le port {port}")
    except Exception as e:
        print(f"⚠️ Impossible de s'enregistrer dans Consul: {e}")

# Enregistrer au démarrage
register_to_consul("galaxy-classifier", 8000)

# ========== INITIALISATION ==========
# On désactive OpenAPI pour éviter l'erreur avec Pydantic
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Architecture du modèle (96×96)
class GalaxyCNN_6layers_96(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.35)
        
        # Pour 96×96 : après 6 poolings → 1×1
        self.fc1 = nn.Linear(64 * 1 * 1, 128)   # 64 → 128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GalaxyCNN_6layers_96()  # ← Nom corrigé
model.load_state_dict(torch.load("models/modele_galaxies.pth", map_location=device))
model.to(device)
model.eval()

# Transformation pour l'inférence (96×96)
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# Classes (en français)
classes = [
    "Disturbed Galaxies",              # Classe 0
    "Merging Galaxies",                # Classe 1
    "Round Smooth Galaxies",           # Classe 2
    "In-between Round Smooth Galaxies",# Classe 3
    "Cigar Shaped Smooth Galaxies",    # Classe 4
    "Barred Spiral Galaxies",          # Classe 5
    "Unbarred Tight Spiral Galaxies",  # Classe 6
    "Unbarred Loose Spiral Galaxies",  # Classe 7
    "Edge-on Galaxies without Bulge",  # Classe 8
    "Edge-on Galaxies with Bulge"      # Classe 9
]

@app.get("/")
def root():
    return {"message": "API Galaxy Classifier - Prête"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/classes")
def get_classes():
    return {"classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, 1)
    
    classe_predite = classes[predicted.item()]
    confiance = probabilities[0][predicted.item()].item()
    
    return JSONResponse(content={
        "classe": classe_predite,
        "indice_classe": predicted.item(),
        "confiance": round(confiance, 4),
        "probabilites": {classes[i]: round(probabilities[0][i].item(), 4) for i in range(10)}
    })

# ========== ENDPOINTS ASYNCHRONES ==========

@app.post("/predict-async")
async def predict_async(file: UploadFile = File(...)):
    """Version asynchrone : envoie l'image dans RabbitMQ"""
    
    # Lire l'image
    contents = await file.read()
    
    # Générer un ID unique pour cette tâche
    task_id = str(uuid.uuid4())
    
    # Encoder l'image en base64 pour l'envoyer dans RabbitMQ
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    # Créer le message
    message = {
        "task_id": task_id,
        "image_base64": image_base64,
        "timestamp": str(uuid.uuid4())
    }
    
    # Envoyer à RabbitMQ
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='galaxy_predictions', durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key='galaxy_predictions',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # rendre le message persistant
            ))
        
        connection.close()
        
        return {
            "status": "accepted",
            "task_id": task_id,
            "message": "Votre image a été mise en file d'attente. Utilisez /result/{task_id} pour obtenir le résultat."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur RabbitMQ: {str(e)}")


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Récupère le résultat d'une tâche asynchrone"""
    
    # Chemin vers le fichier résultats du worker
    import os
    worker_results_file = "../worker-service/results.json"
    
    # Si le chemin relatif ne fonctionne pas, essaie le chemin absolu
    if not os.path.exists(worker_results_file):
        # Essayer un chemin absolu générique
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        worker_results_file = os.path.join(base_dir, "worker-service", "results.json")
    
    print(f"📁 Recherche du fichier résultats : {worker_results_file}")
    
    if os.path.exists(worker_results_file):
        with open(worker_results_file, 'r') as f:
            results = json.load(f)
            if task_id in results:
                return results[task_id]
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Résultat non disponible encore. Réessayez plus tard."
    }