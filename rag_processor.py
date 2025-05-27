# rag_processor.py
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (opcional, útil para desarrollo local)
load_dotenv()

# --- Configuración ---
# Directorio donde ChromaDB almacenará sus datos.
# Para Code Engine, esto estaría dentro del contenedor.
# Para persistencia real, necesitarías un volumen o almacenar en COS.
CHROMA_DATA_PATH = "chroma_db_data"
COLLECTION_NAME = "documentos_expertos"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Modelo de Sentence Transformers

# Configuración para Watsonx.ai (obtener de variables de entorno)
WX_API_KEY = os.getenv("WX_API_KEY")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
WX_URL = os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com") # Ejemplo, ajusta a tu región
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "ibm/granite-13b-chat-v2") # Ejemplo de modelo

class RAGProcessor:
    def __init__(self):
        print("Inicializando RAGProcessor...")
        # Inicializar el modelo de embeddings
        # Este modelo se descarga la primera vez que se usa.
        # En Code Engine, esto sucederá durante el inicio del contenedor si el modelo no está ya en la imagen.
        print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Modelo de embeddings cargado.")

        # Inicializar el cliente de ChromaDB
        # Usamos un cliente persistente que guarda los datos en CHROMA_DATA_PATH
        print(f"Inicializando ChromaDB en: {CHROMA_DATA_PATH}...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        print("ChromaDB inicializado.")

        # Obtener o crear la colección en ChromaDB
        # Usamos nuestro propio modelo de embeddings en lugar del default de Chroma
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        
        try:
            print(f"Intentando obtener o crear la colección: {COLLECTION_NAME}...")
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.sentence_transformer_ef # Importante pasar la función de embedding
            )
            print(f"Colección '{COLLECTION_NAME}' cargada/creada exitosamente.")
        except Exception as e:
            print(f"Error al obtener/crear la colección: {e}")
            # Podrías querer reintentar o manejar esto de forma más robusta
            # Por ahora, si falla, algunas operaciones no funcionarán.
            self.collection = None


        # Inicializar el modelo LLM de Watsonx.ai
        if WX_API_KEY and WX_PROJECT_ID:
            print(f"Configurando cliente de Watsonx.ai para el proyecto {WX_PROJECT_ID}...")
            self.llm_credentials = {
                "url": WX_URL,
                "apikey": WX_API_KEY
            }
            # Parámetros de generación para el LLM
            self.llm_gen_parms = {
                GenParams.DECODING_METHOD: "greedy", # O "sample"
                GenParams.MAX_NEW_TOKENS: 300,
                GenParams.MIN_NEW_TOKENS: 15,
                GenParams.TEMPERATURE: 0.0, # Ajustar para creatividad (0.0 para más determinista)
                # GenParams.TOP_K: 50, # Descomentar si usas "sample"
                # GenParams.TOP_P: 1   # Descomentar si usas "sample"
            }
            try:
                self.llm = Model(
                    model_id=LLM_MODEL_ID,
                    params=self.llm_gen_parms,
                    credentials=self.llm_credentials,
                    project_id=WX_PROJECT_ID
                )
                print(f"Cliente de Watsonx.ai LLM ({LLM_MODEL_ID}) configurado.")
            except Exception as e:
                print(f"Error al inicializar el modelo LLM de Watsonx.ai: {e}")
                self.llm = None
        else:
            print("Advertencia: No se proporcionaron credenciales de Watsonx.ai (WX_API_KEY, WX_PROJECT_ID). La generación de respuestas LLM no funcionará.")
            self.llm = None

    def _embed_text(self, text_chunks):
        """Genera embeddings para una lista de fragmentos de texto."""
        print(f"Generando embeddings para {len(text_chunks)} chunks...")
        embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=False)
        print("Embeddings generados.")
        return embeddings.tolist() # Chroma espera listas

    def load_and_index_documents(self, documents_path="documents"):
        """
        Carga documentos de texto desde un directorio, los divide en chunks,
        genera embeddings y los almacena en ChromaDB.
        """
        if not self.collection:
            print("Error: La colección de ChromaDB no está disponible.")
            return {"status": "error", "message": "ChromaDB collection not initialized."}

        print(f"Cargando documentos desde: {documents_path}...")
        all_chunks = []
        all_metadatas = []
        all_ids = []
        doc_counter = 0

        # Divisor de texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamaño de cada chunk
            chunk_overlap=100   # Superposición entre chunks
        )

        for filename in os.listdir(documents_path):
            if filename.endswith(".txt"): # Asumimos archivos .txt
                file_path = os.path.join(documents_path, filename)
                print(f"Procesando archivo: {filename}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    chunks = text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({"source": filename, "chunk_index": i})
                        all_ids.append(f"{filename}_doc{doc_counter}_chunk{i}")
                    doc_counter += 1
                    print(f"Archivo {filename} dividido en {len(chunks)} chunks.")

                except Exception as e:
                    print(f"Error procesando el archivo {filename}: {e}")
        
        if not all_chunks:
            print("No se encontraron chunks para indexar.")
            return {"status": "no_documents_processed", "indexed_chunks": 0}

        # Generar embeddings para todos los chunks
        # embeddings = self._embed_text(all_chunks) # ChromaDB lo hará internamente si pasamos la embedding_function

        # Añadir a ChromaDB
        # ChromaDB generará los embeddings usando la `embedding_function` que le pasamos al crear la colección.
        print(f"Añadiendo {len(all_chunks)} chunks a la colección '{COLLECTION_NAME}'...")
        try:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"{self.collection.count()} chunks indexados en ChromaDB.")
            return {"status": "success", "indexed_chunks": self.collection.count()}
        except Exception as e:
            print(f"Error al añadir documentos a ChromaDB: {e}")
            return {"status": "error", "message": f"Error adding to ChromaDB: {str(e)}"}


    def query_documents(self, query_text, n_results=3):
        """
        Busca en ChromaDB los chunks más relevantes para una consulta.
        """
        if not self.collection:
            print("Error: La colección de ChromaDB no está disponible.")
            return []
            
        print(f"Realizando búsqueda en ChromaDB para: '{query_text}'...")
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'] # Incluimos documentos y metadatos
            )
            print(f"Búsqueda completada. Documentos encontrados: {len(results.get('documents', [[]])[0])}")
            # results es un diccionario, los documentos están en results['documents'][0]
            # las metadatas en results['metadatas'][0]
            # las distancias en results['distances'][0]
            
            # Formateamos la salida para que sea más fácil de usar
            retrieved_chunks = []
            if results and results.get('documents') and results.get('documents')[0]:
                for i, doc_content in enumerate(results['documents'][0]):
                    retrieved_chunks.append({
                        "content": doc_content,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {},
                        "distance": results['distances'][0][i] if results.get('distances') and results['distances'][0] else float('inf')
                    })
            return retrieved_chunks
        except Exception as e:
            print(f"Error durante la búsqueda en ChromaDB: {e}")
            return []

    def generate_answer_with_llm(self, question, retrieved_chunks):
        """
        Genera una respuesta usando el LLM de Watsonx.ai,
        basándose en la pregunta y los chunks recuperados.
        """
        if not self.llm:
            print("Advertencia: El cliente LLM de Watsonx.ai no está inicializado. Devolviendo respuesta genérica.")
            return "El servicio LLM no está configurado. No se puede generar una respuesta."

        if not retrieved_chunks:
            context_str = "No se encontró información relevante en los documentos."
        else:
            context_str = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])

        prompt_template = f"""Eres un asistente experto que responde preguntas basándose únicamente en el siguiente contexto. Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente información.

Contexto:
{context_str}

Pregunta: {question}

Respuesta:
"""
        print("\n--- Prompt Enviado al LLM ---")
        print(prompt_template)
        print("---------------------------\n")

        try:
            print("Generando respuesta con Watsonx.ai LLM...")
            response = self.llm.generate_text(prompt=prompt_template)
            print("Respuesta recibida del LLM.")
            return response
        except Exception as e:
            print(f"Error al generar respuesta con Watsonx.ai LLM: {e}")
            return f"Error al contactar el servicio LLM: {str(e)}"

# Para pruebas locales (opcional)
if __name__ == "__main__":
    # Crear un directorio de documentos si no existe y añadir archivos de ejemplo
    if not os.path.exists("documents"):
        os.makedirs("documents")
    with open("documents/doc1.txt", "w") as f:
        f.write("IBM Code Engine es una plataforma serverless que permite ejecutar contenedores, aplicaciones y funciones. Es ideal para desplegar aplicaciones web y APIs.")
    with open("documents/doc2.txt", "w") as f:
        f.write("ChromaDB es una base de datos vectorial de código abierto diseñada para almacenar y buscar embeddings. Se integra bien con modelos de lenguaje grandes para aplicaciones RAG.")
    with open("documents/doc3.txt", "w") as f:
        f.write("Watsonx.ai es la plataforma de inteligencia artificial de IBM que ofrece modelos fundacionales y herramientas MLOps para construir y desplegar aplicaciones de IA a escala.")

    # Asegúrate de tener tus variables de entorno WX_API_KEY y WX_PROJECT_ID configuradas en un archivo .env o en el sistema
    # para que la parte del LLM funcione.
    
    processor = RAGProcessor()
    
    # Indexar documentos (solo es necesario hacerlo una vez o cuando los documentos cambian)
    print("\n--- Indexando Documentos ---")
    index_result = processor.load_and_index_documents()
    print(f"Resultado de la indexación: {index_result}")

    # Realizar una consulta
    print("\n--- Consultando Documentos ---")
    user_question = "¿Qué es IBM Code Engine?"
    chunks = processor.query_documents(user_question, n_results=2)
    
    if chunks:
        print(f"Chunks recuperados para '{user_question}':")
        for i, chunk_data in enumerate(chunks):
            print(f"  Chunk {i+1} (Fuente: {chunk_data['metadata'].get('source', 'N/A')}, Distancia: {chunk_data['distance']:.4f}):")
            print(f"    '{chunk_data['content'][:100]}...'") # Mostrar solo los primeros 100 caracteres
    else:
        print(f"No se encontraron chunks relevantes para '{user_question}'.")

    # Generar respuesta con LLM (si está configurado)
    if processor.llm and chunks:
        print("\n--- Generando Respuesta con LLM ---")
        answer = processor.generate_answer_with_llm(user_question, chunks)
        print(f"\nPregunta: {user_question}")
        print(f"Respuesta del LLM: {answer}")
    elif not processor.llm:
        print("\nLLM no configurado, no se puede generar respuesta.")
    elif not chunks:
         print("\nNo se recuperaron chunks, no se puede generar respuesta con contexto.")

    # Otra pregunta
    print("\n--- Consultando Documentos (otra pregunta) ---")
    user_question_2 = "¿Cómo funciona ChromaDB con RAG?"
    chunks_2 = processor.query_documents(user_question_2, n_results=2)
    if chunks_2:
        print(f"Chunks recuperados para '{user_question_2}':")
        for i, chunk_data in enumerate(chunks_2):
            print(f"  Chunk {i+1} (Fuente: {chunk_data['metadata'].get('source', 'N/A')}, Distancia: {chunk_data['distance']:.4f}):")
            print(f"    '{chunk_data['content'][:100]}...'")
    else:
        print(f"No se encontraron chunks relevantes para '{user_question_2}'.")
    
    if processor.llm and chunks_2:
        print("\n--- Generando Respuesta con LLM (otra pregunta) ---")
        answer_2 = processor.generate_answer_with_llm(user_question_2, chunks_2)
        print(f"\nPregunta: {user_question_2}")
        print(f"Respuesta del LLM: {answer_2}")