import os
from pathlib import Path
from typing import List
import llama_index
from pydantic import BaseModel, json
from pathlib import Path 
from llama_index import download_loader
from llama_index.llms import OpenAI
from llama_index.llms import AzureOpenAI
from llama_index import VectorStoreIndex, ServiceContext
from dotenv import load_dotenv
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index.readers.docx_reader import DocXReader

#from llama_index.readers.docx_reader import DocXReader



import openai
from collections import Counter
import pandas as pd
import sys
sys.path.append ('/path/to/llama_index')

PptxReader = download_loader("PptxReader")
PDFMinerReader = download_loader("PDFMinerReader")
DocXReader = download_loader("DocXReader")

docxloader = DocXReader()
pptxloader = PptxReader()
loader = PDFMinerReader()

load_dotenv()
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = os.getenv("OPENAI_API_VERSION")
OPENAI_DEVINCI_MODEL = os.getenv("OPENAI_DEVINCI_MODEL")
engine = os.getenv("engine")

llm = AzureOpenAI(
    engine = engine,
    model = engine,
    temperature = 0.0,
    azure_endpoint = openai.api_base,
    api_key = openai.api_key,
    api_version = openai.api_version,
)

embed_model = AzureOpenAIEmbedding(
    model = "text-embedding-ada-002",
    deployment_name = "text-embedding-ada-002",
    api_key = openai.api_key,
    azure_endpoint = openai.api_base,
    api_version = openai.api_version,
)

service_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model = embed_model,
)

set_global_service_context(service_context)

def load_keywords_from_csv(file_name: str) -> List[str]:
    df = pd.read_csv(file_name, header = None)
    keywords = df[0].tolist()
    return keywords

csv_file_name = "tags.csv"
pre_existing_keywords = load_keywords_from_csv(csv_file_name)


def find_matching_keywords(file_content: str, pre_existing_keywords: List[str], num_keywords: int = 3) -> List[str]:
    #split file content into words
    words = file_content.split()

    #count frequency of each pre-existing keyword in the file content
    keyword_counts = Counter({kw: words.count(kw) for kw in pre_existing_keywords})

    #sort ore-existing keywords by their frequency
    sorted_keywords = [kw for kw, _ in keyword_counts.most_common()]

    #find matching keywords
    matching_keywords = [kw for kw in sorted_keywords if kw in words]

    #return most frequent keyword if no matching
    if not matching_keywords:
        return [sorted_keywords[0]]
    
    #retur top matching keywords
    return matching_keywords[:num_keywords]


#directory containing pdf files
directory_path = "./uploads"

#list all files in the directory
files = os.listdir(directory_path)
file_dict = dict()

output_data = pd.DataFrame(columns = ["File Name", "Keywords"])


#iterate through the names of the file
for file in files:
    file_path = os.path.join(directory_path, file)
    if file_path.endswith('.pdf'):
        try:
            #create a pdfMiner instance
            doc = loader.load_data(Path(file_path))
            print('success' + file_path)

        except Exception as e:
            print('failure' + file_path)
            print("Error:", e)
            continue

    elif file_path.endswith('.pptx'):
        try:
            doc = pptxloader.load_data(file_path)
            print('success' + file_path)

        except Exception as e:
            print('failure' + file_path)
            print("Error:", e)
            continue

    elif file_path.endswith('.docx'):
        try:
            doc = docxloader.load_data(Path(file_path))
            print('success' + file_path)

        except Exception as e:
            print('failure' + file_path)
            print("Error", e)
            continue

    index = VectorStoreIndex.from_documents(doc, service_context = service_context)

    query_engine = index.as_query_engine()

    response = query_engine("Can you help summarize this document")
    print(response)

    #get the content of the document as a string
    file_content = " ".join(page.text for page in doc)

    #find matching keywords
    matching_keywords = find_matching_keywords(file_content, pre_existing_keywords)

    #print the matching keywords for the current file
    print("Matching keywords for{file}: {matching_keywords}")


#write the DataFrame to an Excel file
    output_file = "output_results.xlsx"
    output_data.to_excel(output_file, index = False)


