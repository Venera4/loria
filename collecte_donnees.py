# Génère un dataset synthétique SHS avec Claude (Anthropic)

import pandas as pd
import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Charger les variables d'environnement (.env avec ANTHROPIC_API_KEY)
load_dotenv()

# Initialiser le client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

model = "claude-sonnet-4-6" 

#  Labels
labels_config = {
    "cinéma": 39,
    "musique": 42,
    "théâtre": 25,
    "peinture": 26,
    "sculpture": 28,
    "littérature": 34,
    "danse": 29,
    "architecture": 19,
    "photographie": 22,
    "cinéma d’animation": 7,
    "critique culturelle": 10,
    "patrimoine": 33,
    "design": 25,
    "mode": 17
}

csv_file = "synthetic_SHS_dataset.csv"

# Reset CSV
if os.path.exists(csv_file):
    os.remove(csv_file)

batch_size = 5  # Claude = plus cher → batch plus petit

# 2 Génération
for label, n_texts in labels_config.items():
    print(f" Génération de {n_texts} textes pour '{label}'...")

    for i in range(0, n_texts, batch_size):
        current_batch = min(batch_size, n_texts - i)

        prompt = f"""
Génère {current_batch} courts textes en français sur le thème culturel '{label}'.
Chaque texte doit être autonome, comme un extrait d'article ou critique culturelle.
Les textes doivent être variés et réalistes.
Renvoie chaque texte séparé par une ligne vide.
Chaque texte doit être indépendant. Ne numérote PAS les textes (pas de "1.", "2.", etc.)
Ne mets PAS de puces ou tirets ou de apostrophes ou guillemets. Juste du texte brut.
"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=800,
                temperature=0.9,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            text = response.content[0].text

            # Split
            batch_texts = [t.strip() for t in text.split("\n\n") if t.strip()]
            batch_texts = batch_texts[:current_batch]

            # DataFrame
            df_batch = pd.DataFrame({
                "text": batch_texts,
                "label": [label]*len(batch_texts)
            })

            # Write CSV
            if not os.path.exists(csv_file):
                df_batch.to_csv(csv_file, index=False, encoding="utf-8", mode='w')
            else:
                df_batch.to_csv(csv_file, index=False, encoding="utf-8", mode='a', header=False)

            print(f" Batch {len(batch_texts)} ajouté")

        except Exception as e:
            print(f" Erreur : {e}")

print(f" Dataset créé : {csv_file}")