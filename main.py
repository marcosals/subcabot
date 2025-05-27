# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import shutil

from rag_processor import RAGProcessor # Importamos nuestra clase

# --- Modelos de Datos Pydantic para las Solicitudes y Respuestas ---
class AskRequest(BaseModel):
    question: str
    n_results: int = 3 # Número de chunks a recuperar por defecto

class AskResponse(BaseModel):
    question: str
    retrieved_chunks: list
    llm_answer: str
    
class IndexResponse(BaseModel):
    status: str
    message: str
    indexed_chunks: int = 0

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str


# --- Inicialización de la Aplicación FastAPI y el Procesador RAG ---
app = FastAPI(
    title="API RAG con ChromaDB y Watsonx.ai",
    description="Una API para realizar búsquedas semánticas en documentos y generar respuestas con un LLM.",
    version="0.1.0"
)

# Creamos una instancia global de nuestro RAGProcessor.
# Esto asegura que el modelo de embeddings y ChromaDB se carguen solo una vez al iniciar la app.
# En un entorno serverless como Code Engine, esto podría suceder en cada "arranque en frío" de una instancia.
rag_processor_instance = RAGProcessor()

# Directorio para almacenar documentos subidos temporalmente
UPLOAD_DIR = "documents_uploaded" 
# Directorio que usará rag_processor para indexar (puede ser el mismo o diferente)
DOCUMENTS_DIR = "documents" 

# Asegurarse de que los directorios existan
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# --- Endpoints de la API ---

@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta al iniciar la aplicación.
    Podríamos precargar/indexar documentos aquí si es necesario,
    aunque es mejor tener un endpoint dedicado para ello.
    """
    print("Aplicación FastAPI iniciada.")
    # Verificar si hay documentos en DOCUMENTS_DIR y si la colección está vacía
    # Esto es una lógica simple, podría mejorarse
    if rag_processor_instance.collection and rag_processor_instance.collection.count() == 0:
        print("La colección de ChromaDB está vacía. Intentando indexar documentos existentes en 'documents/'...")
        if any(fname.endswith('.txt') for fname in os.listdir(DOCUMENTS_DIR)):
            result = rag_processor_instance.load_and_index_documents(documents_path=DOCUMENTS_DIR)
            print(f"Resultado de la indexación automática al inicio: {result}")
        else:
            print("No se encontraron documentos en 'documents/' para la indexación automática.")


@app.post("/upload_document/", response_model=UploadResponse)
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo de texto al directorio 'documents'.
    Después de subir, se debería llamar a /build_index/ para procesarlo.
    """
    # Por seguridad, solo permitimos archivos .txt
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt")
    
    file_path = os.path.join(DOCUMENTS_DIR, file.filename) # Guardar en el directorio que usa el indexador
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return UploadResponse(
            status="success", 
            message=f"Archivo '{file.filename}' subido exitosamente. Recuerda re-indexar.",
            filename=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")


@app.post("/build_index/", response_model=IndexResponse)
async def build_index_endpoint():
    """
    Endpoint para (re)construir el índice de ChromaDB con los documentos
    que se encuentren en el directorio 'documents'.
    """
    print("Solicitud para construir/reconstruir el índice recibida.")
    try:
        # Antes de indexar, podríamos querer limpiar la colección si es una re-indexación completa.
        # Sin embargo, la lógica actual de RAGProcessor.load_and_index_documents
        # añade documentos. Si los IDs son los mismos, ChromaDB los actualiza.
        # Si queremos una limpieza total, tendríamos que borrar y recrear la colección.
        # Por simplicidad, ahora solo añadimos/actualizamos.
        
        # Opcional: Limpiar la colección antes de reindexar todo
        # if rag_processor_instance.collection:
        #     print(f"Limpiando la colección '{rag_processor_instance.COLLECTION_NAME}' antes de reindexar...")
        #     rag_processor_instance.chroma_client.delete_collection(name=rag_processor_instance.COLLECTION_NAME)
        #     rag_processor_instance.collection = rag_processor_instance.chroma_client.get_or_create_collection(
        #         name=rag_processor_instance.COLLECTION_NAME,
        #         embedding_function=rag_processor_instance.sentence_transformer_ef
        #     )
        #     print("Colección limpiada y recreada.")

        result = rag_processor_instance.load_and_index_documents(documents_path=DOCUMENTS_DIR)
        if result["status"] == "success" or result["status"] == "no_documents_processed":
            return IndexResponse(
                status=result["status"],
                message="Índice construido/actualizado exitosamente." if result["status"] == "success" else "No se procesaron nuevos documentos.",
                indexed_chunks=result.get("indexed_chunks", 0)
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Error desconocido durante la indexación."))
    except Exception as e:
        print(f"Error en el endpoint /build_index/: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al construir el índice: {str(e)}")


@app.post("/ask/", response_model=AskResponse)
async def ask_question_endpoint(request: AskRequest):
    """
    Endpoint para hacer una pregunta. La pregunta se usará para buscar
    chunks relevantes en ChromaDB, y luego se pasará al LLM para generar una respuesta.
    """
    print(f"Pregunta recibida: '{request.question}', n_results: {request.n_results}")
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        # 1. Recuperar chunks relevantes de ChromaDB
        retrieved_chunks = rag_processor_instance.query_documents(request.question, n_results=request.n_results)
        
        if not retrieved_chunks:
            print("No se encontraron chunks relevantes.")
            # Aún así, podríamos intentar preguntar al LLM sin contexto o devolver un mensaje específico.
            # Por ahora, si no hay chunks, el LLM indicará que no tiene contexto.
            pass

        # 2. Generar respuesta con el LLM usando los chunks recuperados
        llm_answer = rag_processor_instance.generate_answer_with_llm(request.question, retrieved_chunks)

        return AskResponse(
            question=request.question,
            retrieved_chunks=retrieved_chunks, # Devolvemos los chunks para transparencia/depuración
            llm_answer=llm_answer
        )
    except Exception as e:
        print(f"Error en el endpoint /ask/: {e}")
        # Considerar no exponer detalles internos del error al cliente en producción
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar la pregunta: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API RAG. Usa los endpoints /build_index/ y /ask/."}

# Para ejecutar con Uvicorn localmente:
# uvicorn main:app --reload
