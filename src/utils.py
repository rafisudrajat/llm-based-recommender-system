import os
from openai import AzureOpenAI
from mimetypes import guess_type
import base64
from PIL import ImageFile
from io import BytesIO
from dotenv import load_dotenv
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
load_dotenv()

API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
API_KEY = os.getenv("AZURE_OPENAI_KEY")

LLM_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
LLM_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") # this might change in the future

EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT")
EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION") # this might change in the future

def create_azure_openai_llm_client()->AzureOpenAI:
    return AzureOpenAI(
        api_key=API_KEY,  
        api_version=LLM_API_VERSION,
        azure_endpoint=API_BASE,
    )

def create_azure_openai_embedding_client()->AzureOpenAI:
    return AzureOpenAI(
        api_key=API_KEY,  
        api_version=EMBEDDING_API_VERSION,
        azure_endpoint=API_BASE,
    )

def read_and_format_user_data()->dict[int,dict[str,str|list[str]]]:
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the absolute path to the JSON file
    file_path = os.path.join(script_dir, '..', 'data', 'people_data.json')
    formatted_user_data = {}
    with open(file_path) as f:
        user_data = json.load(f)
        for user in user_data:
            if formatted_user_data.get(user['user_id'],None) == None:
                copy_user = user.copy()
                del copy_user['user_id']
                formatted_user_data[user['user_id']]=copy_user
        return formatted_user_data        

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path:str)->str:
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def convert_PIL_image_to_data_url(image:ImageFile,content_type:str)->str:
    media_type = content_type.split('/')[1]
    im_file = BytesIO()
    image.save(im_file, format=media_type)
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    # Construct the data URL
    return f"data:{content_type};base64,{im_b64}"


def create_milvus_collection(collection_name:str, dim:int, drop_existing_collection:bool=False):
    connections.connect(host='0.0.0.0', port='19530')
    if drop_existing_collection and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),   
        FieldSchema(name="name_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="steps", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="ingredients", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="ingredients_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="synthetic_review", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="synthetic_review_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description='Food recipe data')
    collection = Collection(name=collection_name, schema=schema,enable_dynamic_field=False)
    
    vector_index_params = {
        'metric_type': "COSINE",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 240}
    }
    scalar_index_params = {
        'index_name': "name_index",
        "index_type":"INVERTED"
    }
    collection.create_index(field_name='name_embedding', index_params=vector_index_params)
    collection.create_index(field_name='ingredients_embedding', index_params=vector_index_params)
    collection.create_index(field_name='synthetic_review_embedding', index_params=vector_index_params)
    collection.create_index(field_name='name', index_params=scalar_index_params)

    return collection