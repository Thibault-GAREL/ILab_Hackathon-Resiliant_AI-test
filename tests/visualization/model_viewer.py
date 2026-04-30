from transformers import AutoConfig, AutoModelForCausalLM
import torch

model_id = "google/gemma-4-E2b"

# 1. On charge juste la "recette" (très léger)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("--- CONFIGURATION DU MODÈLE ---")
print(config)

# 2. On charge le modèle SANS les poids (avec des valeurs aléatoires/vides)
# Cela permet de voir la structure exacte sans consommer 10Go de RAM
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# # Pour voir uniquement la tour de Vision
# print(model.model.vision_tower)

# # Pour voir uniquement les couches de décodage de texte
# print(model.model.language_model.layers)

# # Pour voir comment le modèle passe de la Vision au Texte (le "projecteur")
# if hasattr(model.model, "vision_projector"):
#     print(model.model.vision_projector)

# On convertit la structure du modèle en chaîne de caractères
architecture_string = str(model)

# On l'écrit dans un fichier texte
with open("vizualisation_document/architecture_gemma4.txt", "w", encoding="utf-8") as f:
    f.write(architecture_string)

print("L'architecture complète a été sauvegardée dans 'architecture_gemma4.txt'")

# Optionnel : Utiliser torchinfo pour un résumé plus détaillé (sans les poids)
from torchinfo import summary

model_stats = summary(model, verbose=0, depth=4)  # depth=4 pour voir les sous-couches

with open("vizualisation_document/summary_gemma4.txt", "w", encoding="utf-8") as f:
    f.write(str(model_stats))

# On sauvegarde aussi l'architecture brute pour avoir le nom exact des variables
with open("vizualisation_document/architecture_brute_gemma4.txt", "w", encoding="utf-8") as f:
    f.write(str(model))
