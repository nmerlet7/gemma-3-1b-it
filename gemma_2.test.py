from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import os

app = Flask(__name__, static_folder=".", static_url_path="")

# Clé et modèle
HUGGING_FACE_API_KEY = "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"
model_id = "google/gemma-3-1b-it"

# Chargement du modèle et tokenizer
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    use_auth_token=HUGGING_FACE_API_KEY
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=HUGGING_FACE_API_KEY
)

@app.route("/")
def index():
    template_folder = os.path.join(app.root_path, 'templates')
    file_path = os.path.join(template_folder, 'index.html')

    if not os.path.exists(file_path):
        print("❌ ERREUR : Le fichier index.html est introuvable dans le dossier templates.")
        return "<h1>Erreur : fichier index.html non trouvé.</h1>", 404

    print(f"➡️ Tentative d'envoi de : {file_path}")
    return send_from_directory(template_folder, "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_input = data.get("text", "")

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_input}]
            }
        ]
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device).to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True
        )

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return jsonify({"reply": decoded_output[0]})

if __name__ == "__main__":
    app.run(debug=True)