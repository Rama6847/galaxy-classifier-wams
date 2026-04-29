import pika
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import base64
from datetime import datetime
import consul
import socket

import consul
import socket
from fastapi import FastAPI
import threading
import uvicorn

def register_to_consul(service_name: str, port: int):
    try:
        c = consul.Consul(host='localhost', port=8500)
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
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

# Serveur HTTP pour les health checks
worker_app = FastAPI()

@worker_app.get("/health")
def health():
    return {"status": "ok"}

def run_health_server():
    uvicorn.run(worker_app, host="0.0.0.0", port=8002, log_level="warning")

# Démarrer le serveur
health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()

# Enregistrer dans Consul
register_to_consul("galaxy-worker", 8002)

# ========== CONFIGURATION ==========
# ========== CONFIGURATION ==========
RABBITMQ_HOST = "localhost"
QUEUE_NAME = "galaxy_predictions"


import os
# Remonte d'un dossier pour être dans worker-service/ (pas dans app/)
WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(WORKER_DIR, "results.json")

print(f"📁 Les résultats seront sauvegardés dans : {RESULTS_FILE}")

# ========== DÉFINITION DU MODÈLE ==========
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
        self.fc1 = nn.Linear(64, 128)
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

# ========== CHARGEMENT DU MODÈLE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GalaxyCNN_6layers_96()

# Chercher le modèle
model_path = "../models/modele_galaxies.pth"
if not os.path.exists(model_path):
    model_path = "models/modele_galaxies.pth"
if not os.path.exists(model_path):
    model_path = "C:/Users/Khodja Rahma/Desktop/etudes_universiteres/Master/M1/S2/Projet/Deploiement/wams_galaxy_project/models/modele_galaxies.pth"

print(f"📦 Chargement du modèle depuis {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Modèle chargé avec succès")

# ========== TRANSFORMATION ==========
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# ========== CLASSES ==========
classes = [
    "Galaxies perturbées", "Galaxies en fusion", "Galaxies lisses rondes",
    "Galaxies lisses intermédiaires", "Galaxies lisses en cigare",
    "Galaxies spirales barrées", "Galaxies spirales serrées",
    "Galaxies spirales lâches", "Galaxies vues par la tranche sans bulbe",
    "Galaxies vues par la tranche avec bulbe"
]

# ========== FONCTIONS DE STOCKAGE ==========
def save_result(task_id: str, result: dict):
    """Sauvegarde le résultat dans un fichier JSON"""
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    
    results[task_id] = result
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Résultat sauvegardé pour {task_id}")

def get_result(task_id: str):
    """Récupère un résultat depuis le fichier JSON"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
            return results.get(task_id)
    return None

# ========== FONCTION DE PRÉDICTION ==========
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, 1)
    
    return {
        "classe": classes[predicted.item()],
        "indice_classe": predicted.item(),
        "confiance": round(probabilities[0][predicted.item()].item(), 4),
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }

# ========== CONNEXION À RABBITMQ ==========
def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        task_id = message.get("task_id")
        image_base64 = message.get("image_base64")
        image_bytes = base64.b64decode(image_base64)
        
        print(f"🔄 Traitement de la tâche {task_id}...")
        
        result = predict_image(image_bytes)
        result["task_id"] = task_id
        
        save_result(task_id, result)
        
        print(f"✅ Tâche {task_id} terminée : {result['classe']} (confiance: {result['confiance']})")
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

# ========== DÉMARRAGE ==========
def start_worker():
    print("🚀 Démarrage du worker...")
    print(f"📡 Connexion à RabbitMQ sur {RABBITMQ_HOST}")
    
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    print(f"✅ Worker en écoute sur la queue '{QUEUE_NAME}'")
    print("🎧 En attente de messages... (Ctrl+C pour arrêter)")
    
    channel.start_consuming()

if __name__ == "__main__":
    start_worker()