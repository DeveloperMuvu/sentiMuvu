import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm  # Para la barra de progreso
import matplotlib.pyplot as plt  # Para graficar
from collections import Counter  # Para contar palabras
import re  # Para expresiones regulares
import requests

# Carga de datos desde un archivo Excel
df = pd.read_excel("encuesta.xlsx")

# Verificar si CUDA está disponible para usar GPU
device = 0 if torch.cuda.is_available() else -1  # Usar GPU si está disponible, de lo contrario usar CPU

# Configuración del pipeline de análisis de sentimiento
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)



def analizar_sentimiento_sentigem(texto):
    api_key = ''  # Reemplaza con tu clave API
    url = 'https://api.sentigem.com/external/get-sentiment'
    
    # Parámetros para la solicitud
    params = {
        'api-key': api_key,
        'text': texto,
    }
    
    try:
        # Realizar la solicitud GET
        response = requests.get(url, params=params,verify=False)
        response_data = response.json()  # Convertir la respuesta a JSON
        
        # Verificar el estado de la respuesta
        if response_data['status'] == "1":
            return response_data['polarity']  # Retorna el sentimiento (positivo, negativo, neutro)
        else:
            print(f"Error: {response_data['message']}")
            return None
            
    except Exception as e:
        print(f"Error en la conexión: {e}")
        return None


# Función para clasificar el sentimiento en positiva, negativa o neutra
def analizar_sentimiento(texto):
    if not isinstance(texto, str):  # Verificar que el texto sea una cadena
        return None  # O manejarlo como prefieras
    
    try:
        resultado = sentiment_analyzer(texto[:512])  # Limitar el texto a 512 caracteres
        if isinstance(resultado, list) and len(resultado) > 0:
            label = resultado[0]['label']
            
            # Clasificación según el resultado
            if label in ["5 stars", "4 stars"]:
                return "Positiva"
            elif label == "3 stars":
                return "Neutra"
            else:
                return "Negativa"
        else:
            return None  # Manejar caso donde no hay resultados válidos
    except Exception as e:
        print(f"Error al analizar el texto: {e}")
        return None  # O manejarlo como prefieras

# Función para calcular la longitud de las respuestas
def sentence_len(texto):
    if isinstance(texto, str):  # Verificar que el texto sea una cadena
        return len(texto.split())
    else:
        return 0  # Retornar 0 si no es una cadena

# Aplicar análisis de sentimiento en la columna específica con barra de progreso
tqdm.pandas(desc="Analizando sentimientos")  # Configurar tqdm para pandas
df["nuevo_Sentimiento"] = df["Cuéntanos en qué podemos mejorar."].progress_apply(analizar_sentimiento)
#df["nuevo_Sentimiento_sentigen"] = df["Cuéntanos en qué podemos mejorar."].progress_apply(analizar_sentimiento_sentigem)

# Aplicar la función para calcular la longitud de las respuestas
df['longitud_respuesta'] = df['Cuéntanos en qué podemos mejorar.'].apply(sentence_len)

# Guardar el resultado en un nuevo archivo Excel
df.to_excel("archivo_resultado.xlsx", index=False)
print("Análisis completado y guardado en archivo_resultado.xlsx")

# Contar las palabras más comunes en los comentarios
def contar_palabras(comentarios):
    palabras = []
    for comentario in comentarios:
        if isinstance(comentario, str):  # Asegurarse de que sea una cadena
            # Limpiar el texto: eliminar puntuación y convertir a minúsculas
            comentario = re.sub(r'[^\w\s]', '', comentario.lower())
            palabras.extend(comentario.split())
    
    return Counter(palabras)

# Contar palabras y obtener las 10 más comunes
contador_palabras = contar_palabras(df["Cuéntanos en qué podemos mejorar."])
palabras_comunes = contador_palabras.most_common(10)

# Separar las palabras y sus conteos para graficar
palabras, conteos = zip(*palabras_comunes)

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(palabras, conteos, color='skyblue')
plt.title('10 Palabras Más Comunes en Comentarios')
plt.xlabel('Palabras')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.tight_layout()  # Ajustar el diseño para que no se corten las etiquetas

# Mostrar el gráfico
plt.show()