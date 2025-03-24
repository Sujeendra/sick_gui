import os
import warnings
import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from rag_framework import *


def process_chat(prompt):
        # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress LangChain deprecation warnings

    # Suppress logs
    logging.getLogger("langchain").setLevel(logging.ERROR)

    # Configure the environment variable to address the OpenMP duplicate library conflict.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load environment variables
    load_dotenv()

    # Load API key
    os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "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")
    UPLOAD_DIR = r"/home/sujee/sick_gui/scripts/sick_mine_llm/local_database"

    # Ensure the UPLOAD_DIR directory exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Load documents and create vector store
    docs = upload_data()
    create_vector_store(docs)

    # Initialize chat history
    chat_history = []

    # Create LLM model
    llm = create_llm_model()

    # Load embeddings and FAISS index
    embeddings = NVIDIAEmbeddings()
    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


    

    # Get response from the model
    response = get_response(llm=llm, 
                            vector_DB=faiss_index, 
                            question=prompt, 
                            chat_history=chat_history)
    return response["answer"]

