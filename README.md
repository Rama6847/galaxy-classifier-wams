# Galaxy Classifier - Projet WAMS

Application de classification de galaxies basée sur le Deep Learning, déployée avec une architecture microservices.

## 🚀 Architecture

- **API Galaxy Classifier** (port 8000) : Classification des galaxies (PyTorch)
- **Auth Service** (port 8001) : Authentification JWT et gestion des rôles
- **Worker Service** (port 8002) : Traitement asynchrone via RabbitMQ
- **UI Frontend** (port 8080) : Interface web utilisateur
- **Traefik** (port 80) : Reverse proxy et load balancer
- **Consul** (port 8500) : Service discovery
- **RabbitMQ** (port 5672) : File de messages asynchrone

## 📦 Installation

```bash
git clone https://github.com/Rama6847/galaxy-classifier-wams.git
cd galaxy-classifier-wams
python -m venv project_env
.\project_env\Scripts\Activate.ps1
pip install -r requirements.txt
