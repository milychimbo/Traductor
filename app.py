from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Modelos disponibles para traducción
ModelosIdioma = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en"
}

# Modelo de traducción de español a quechua
modelo_quechua = "somosnlp-hackathon-2022/t5-small-finetuned-spanish-to-quechua"
modelo_quechua_t5 = AutoModelForSeq2SeqLM.from_pretrained(modelo_quechua)
tokenizer_quechua_t5 = AutoTokenizer.from_pretrained(modelo_quechua)

def traducir_a_quechua(texto):
    """Traduce texto de español a quechua utilizando el modelo T5."""
    input_ids = tokenizer_quechua_t5(texto, return_tensors="pt").input_ids
    outputs = modelo_quechua_t5.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)
    texto_traducido = tokenizer_quechua_t5.decode(outputs[0], skip_special_tokens=True)
    return texto_traducido

@app.route('/traducir', methods=['POST'])
def traducir_texto():
    try:
        data = request.get_json()
        texto = data.get('text')
        idioma_origen = data.get('source_lang', 'es')
        idioma_destino = data.get('target_lang', 'en')

        if not texto:
            return jsonify({'error': 'No se proporcionó texto para traducir'}), 400

        # Si la traducción es español → quechua
        if idioma_origen == "es" and idioma_destino == "qu":
            texto_traducido = traducir_a_quechua(texto)
            return jsonify({'translated_text': texto_traducido})

        # Si el idioma está en la lista de modelos de Helsinki-NLP
        modelo_hf = ModelosIdioma.get((idioma_origen, idioma_destino))
        if modelo_hf:
            traductor = pipeline("translation", model=modelo_hf)
            texto_traducido = traductor(texto)[0]['translation_text']
            return jsonify({'translated_text': texto_traducido})

        return jsonify({'error': f'La traducción de {idioma_origen} a {idioma_destino} no está soportada'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
