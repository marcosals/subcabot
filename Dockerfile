# Usar una imagen base de Python oficial
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt requirements.txt

# Instalar las dependencias
# --no-cache-dir para reducir el tamaño de la imagen
# --default-timeout=300 para dar más tiempo a la descarga de paquetes grandes
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copiar el resto del código de la aplicación al directorio de trabajo
COPY . .

# Variables de entorno para Watsonx.ai (se deben configurar en Code Engine)
# ENV WX_API_KEY="tu_api_key_aqui"
# ENV WX_PROJECT_ID="tu_project_id_aqui"
# ENV WX_URL="https://tu.region.ml.cloud.ibm.com"
# ENV LLM_MODEL_ID="ibm/granite-13b-chat-v2" # o el modelo que prefieras

# Exponer el puerto en el que Uvicorn se ejecutará (Code Engine lo detectará)
EXPOSE 8080

# Comando para ejecutar la aplicación FastAPI con Uvicorn
# Code Engine espera que la aplicación escuche en el puerto definido por la variable PORT (por defecto 8080)
# Usamos 0.0.0.0 para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
