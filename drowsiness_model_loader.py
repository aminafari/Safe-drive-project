import torch
import torch.nn as nn
from torchvision import models

def load_drowsiness_model(path, device):
    try:
        # ⚠️ IMPORTANT : même architecture que training
        model = models.resnet18(weights=None)

        # ⚠️ IMPORTANT : même nombre de classes
        num_classes = 2  # adapte si besoin
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Charger les poids
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)  # strict=True par défaut ✅

        model = model.to(device)
        model.eval()

        print("✅ Modèle chargé correctement (100%)")
        return model, True

    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None, False