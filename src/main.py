from typing import Union
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
import io
from .utils import create_azure_openai_llm_client, create_azure_openai_embedding_client, convert_PIL_image_to_data_url, read_and_format_user_data
from .inference import food_analyzer_inference, food_recommendation_inference
from .query_db import find_food_data_by_name

app = FastAPI()

llm_client = create_azure_openai_llm_client()
embedding_client = create_azure_openai_embedding_client()

class RecommendationRequest(BaseModel):
    user_id: int

class SearchFoodRequest(BaseModel):
    recipe_name: str

# convert user data
USER_DATA = read_and_format_user_data()

@app.post("/recipe_info")
async def search_food_by_name(request:SearchFoodRequest):
    recipe_result = find_food_data_by_name(request.recipe_name,
                                           output_fields=["name","ingredients","steps","synthetic_review"],
                                           limit=3)
    return recipe_result


@app.post("/recipe/recommendation")
async def give_food_recommendation(request:RecommendationRequest):
    user_data = USER_DATA.get(request.user_id,None)
    if user_data == None:
        raise HTTPException(status_code=404, detail="User not found")
    response_inference = food_recommendation_inference(llm_client,embedding_client,user_data)
    return {"user_id":response_inference}


@app.post("/food_analyzer/inference")
async def running_food_analyzer(file: UploadFile, user_id: int = Form(...)):
    user_data = USER_DATA.get(user_id,None)
    if user_data == None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Read the file content
    contents = await file.read()
    # Convert to PIL Image
    image = Image.open(io.BytesIO(contents))
    image_url = convert_PIL_image_to_data_url(image,file.content_type)
    response_inference = food_analyzer_inference(llm_client,image_url,user_data)
    return {"response":response_inference}

    