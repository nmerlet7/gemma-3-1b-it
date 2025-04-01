from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
 
# Remplace par ta clé API Hugging Face (⚠️ Ne partage pas ta clé en public !)
HUGGING_FACE_API_KEY = "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"
 
# Identifiant du modèle
model_id = "google/gemma-3-1b-it"
 
# Chargement du modèle sans quantisation
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    use_auth_token=HUGGING_FACE_API_KEY  # Utilisation de la clé API pour l'authentification
).eval()
 
# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HUGGING_FACE_API_KEY)
 
# Message de l'utilisateur et contexte du chatbot
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Vous êtes un assistant utile."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Écris un poème sur Hugging Face, l'entreprise."},]
        },
    ],
]
 # Préparation des entrées pour le modèle
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(torch.bfloat16)

# Génération du texte avec des paramètres personnalisés
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Augmente la longueur du texte généré
        temperature=0.1,  # Réponses plus naturelles et variées
        top_p=0.9,  # Sélectionne les mots avec 90% de probabilité cumulée
        top_k=50,  # Prend les 50 mots les plus probables
        do_sample=True,  # Active l’échantillonnage aléatoire
        repetition_penalty=1.2,  # Réduit les répétitions
        length_penalty=1.0,  # Maintient la longueur standard
        early_stopping=True  # Arrête la génération si une fin de phrase est atteinte
    )

# Décodage et affichage du résultat
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(outputs)