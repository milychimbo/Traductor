import azure.functions as func
import logging
import json
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Crear la aplicación de Azure Functions
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Modelos de traducción
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

@app.function_name(name="traducir")
@app.route(route="traducir", methods=["POST"])
def traducir(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Procesando una solicitud de traducción.')

    try:
        data = req.get_json()

        texto = data.get('text')
        idioma_origen = data.get('source_lang', 'es')
        idioma_destino = data.get('target_lang', 'en')

        if not texto:
            return func.HttpResponse(json.dumps({'error': 'No se proporcionó texto para traducir'}), 
                mimetype="application/json", status_code=400)

        # Traducción español → quechua
        if idioma_origen == "es" and idioma_destino == "qu":
            texto_traducido = traducir_a_quechua(texto)
            return func.HttpResponse(json.dumps({'translated_text': texto_traducido}), 
                mimetype="application/json", status_code=200)

        # Traducción usando modelos Helsinki-NLP
        modelo_hf = ModelosIdioma.get((idioma_origen, idioma_destino))
        if modelo_hf:
            traductor = pipeline("translation", model=modelo_hf)
            texto_traducido = traductor(texto)[0]['translation_text']
            return func.HttpResponse(json.dumps({'translated_text': texto_traducido}), 
                mimetype="application/json", status_code=200)

        # Si los idiomas no están soportados
        return func.HttpResponse(json.dumps({'error': f'La traducción de {idioma_origen} a {idioma_destino} no está soportada'}), 
            mimetype="application/json", status_code=400)

    except ValueError:
        return func.HttpResponse(json.dumps({'error': 'Error en el formato JSON'}), 
            mimetype="application/json", status_code=400)

    except Exception as e:
        return func.HttpResponse(json.dumps({'error': str(e)}), 
            mimetype="application/json", status_code=500)
