# llama_index/__init__.py

# Import submodules
from .llms import OpenAI, AzureOpenAI
from .embeddings import AzureOpenAIEmbedding
from .vector_store import VectorStoreIndex
from .service_context import ServiceContext
from .loader import download_loader

# Define package-level functions or variables if needed
def set_global_service_context(service_context):
    # Your implementation here
    pass

def load_keywords_from_csv(file_name: str) -> List[str]:
    # Your implementation here
    pass

def find_matching_keywords(file_content: str, pre_existing_keywords: List[str], num_keywords: int = 3) -> List[str]:
    # Your implementation here
    pass

# You can define more functions or variables here as needed
