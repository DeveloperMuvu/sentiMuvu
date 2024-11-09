import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = FastAPI()

#tf.config.set_visible_devices([], 'GPU')

# Obtener la API key desde la variable de entorno
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY no está definida en las variables de entorno.")

device = -1

# Configuración del pipeline de análisis de sentimiento
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

# Definición del modelo para la entrada
class SentimentRequest(BaseModel):
    texto: str

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.status_code,
            "message": exc.detail,
        },
    )

@app.post("/analizar_sentimiento/")
def analizar_sentimiento(request: SentimentRequest, x_api_key: str = Header(...)):
    # Validar la API key en la cabecera
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API inválida.")

    texto = request.texto
    
    if not isinstance(texto, str):
        raise HTTPException(status_code=400, detail="Texto inválido.")
    
    try:
        resultado = sentiment_analyzer(texto[:512])  # Limitar el texto a 512 caracteres
        if isinstance(resultado, list) and len(resultado) > 0:
            label = resultado[0]['label']
            if label in ["5 stars", "4 stars"]:
                sentimiento = "Positiva"
            elif label == "3 stars":
                sentimiento = "Neutra"
            else:
                sentimiento = "Negativa"
            return {"sentimiento": sentimiento}
        else:
            raise HTTPException(status_code=500, detail="Error en análisis.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)