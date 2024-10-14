from openai import AzureOpenAI
import os
from .utils import LLM_DEPLOYMENT_NAME, create_milvus_collection
from pymilvus import SearchResult, AnnSearchRequest, WeightedRanker
from .query_db import hybrid_ann_search_for_food_recommendation


def food_analyzer_inference(llm_client:AzureOpenAI,image_url:str,user_info:dict[str,str|list[str]])->str:
    llm_response = llm_client.chat.completions.create(
        model=LLM_DEPLOYMENT_NAME,
        messages=[
            { "role": "system", "content": "You are a helpful assistant that have vast knowledge about food and culinary." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": f"I have this list of food that I cannot eat: {str(user_info['cannot_eat'])}" 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                { 
                    "type": "text", 
                    "text": f"Could you please analyze the image if I can eat that food in the image or not?\
                            Please provide me maximum 4 sentences for the answer." 
                },
            ] } 
        ],
        max_tokens=2000 
    )
    return llm_response.choices[0].message.content

def food_recommendation_inference(llm_client:AzureOpenAI,embedding_client:AzureOpenAI,user_info:dict[str,str|list[str]])->str:
    user_food_preference = user_info["food_preference"]
    
    ingredient_query = parse_food_preference_for_ingredient_query(user_food_preference)
    review_query = parse_food_preference_for_review_query(user_food_preference)
    
    # Convert query string into embedding
    embedding_response = embedding_client.embeddings.create(
            input = [ingredient_query,review_query],
            model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT")
        )
    embedding_response_json = embedding_response.model_dump()
    ingredient_query_embedding = embedding_response_json['data'][0]['embedding']
    review_query_embedding = embedding_response_json['data'][1]['embedding']
    
    # Search similar embedding vector
    hybrid_search_result = hybrid_ann_search_for_food_recommendation(ingredient_query_embedding,review_query_embedding)

    # Filter the food using LLM
    llm_response = llm_client.chat.completions.create(
        model=LLM_DEPLOYMENT_NAME,
        messages=[
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": f"I have a list of food recipe: {hybrid_search_result}" 
                },
                { 
                    "type": "text", 
                    "text": f"And this is a list of food that I cannot eat:{str(user_info['cannot_eat'])}" 
                },
                { 
                    "type": "text", 
                    "text": "From the food recipe list, give me a the name of food that I can eat only! your answer must only contain the food name without ingredients and any additional words" 
                },
                { 
                    "type": "text", 
                    "text": "Example of good result = [egg balado, spicy grilled beef]" 
                }
            ] } 
        ],
        max_tokens=2000 
    )
    return llm_response.choices[0].message.content


# Parser

def parse_food_preference_for_ingredient_query(food_preference:list[str])->str:
    result = ""
    for i, food in enumerate(food_preference):
        if i < len(food_preference)-1:
            result+=f"{food}, "
        else:
            result+=food
    return result

def parse_food_preference_for_review_query(food_preference:list[str])->str:
    result = "Food with "
    for i, food in enumerate(food_preference):
        if i < len(food_preference)-1:
            result+= f"{food} or "
        else:
            result+=food
    return result
