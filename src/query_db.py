from pymilvus import SearchResult, AnnSearchRequest, WeightedRanker
from .utils import create_milvus_collection

# Embedding dimension is 1536
collection = create_milvus_collection("food_recipe_collection",1536,drop_existing_collection=False)

def hybrid_ann_search_for_food_recommendation(ingredient_query_embedding:list[float], review_query_embedding:list[float])->str:
    
    search_param_ingredients = {
        "data": [ingredient_query_embedding], # Query vector
        "anns_field": "ingredients_embedding", # Vector field name
        "param": {
            "metric_type": "COSINE", # This parameter value must be identical to the one used in the collection schema
            "params": {"nprobe": 10}
        },
        "limit": 10 # Number of search results to return in this AnnSearchRequest
    }
    search_request_ingredients = AnnSearchRequest(**search_param_ingredients)

    search_param_food_name = {
        "data": [review_query_embedding], # Query vector
        "anns_field": "name_embedding", # Vector field name
        "param": {
            "metric_type": "COSINE", # This parameter value must be identical to the one used in the collection schema
            "params": {"nprobe": 10}
        },
        "limit": 10 # Number of search results to return in this AnnSearchRequest
    }
    search_request_food_name = AnnSearchRequest(**search_param_food_name)

    search_param_review = {
        "data": [review_query_embedding], # Query vector
        "anns_field": "synthetic_review_embedding", # Vector field name
        "param": {
            "metric_type": "COSINE", # This parameter value must be identical to the one used in the collection schema
            "params": {"nprobe": 10}
        },
        "limit": 10 # Number of search results to return in this AnnSearchRequest
    }
    search_request_review = AnnSearchRequest(**search_param_review)

    # Store these two requests as a list in `reqs`
    search_reqs = [search_request_ingredients, search_request_food_name, search_request_review]
    rerank = WeightedRanker(0.4, 0.2, 0.4)
    
    # Before conducting hybrid search, load the collection into memory.
    collection.load()

    hybrid_search_result = collection.hybrid_search(
        search_reqs, # List of AnnSearchRequests created in step 1
        rerank, # Reranking strategy specified in step 2
        limit=10, # Number of final search results to return,
        output_fields=["name","ingredients","synthetic_review"]
    )

    return parse_db_hybrid_search_result(hybrid_search_result)

def find_food_data_by_name(food_name:str,output_fields:list[str],limit:int=1)->list[dict]:
    # Load the collection into memory.
    collection.load()

    scalar_search_result = collection.query(
        expr=f'name == "{food_name}"',
        limit=limit,
        output_fields=output_fields
    )
    return scalar_search_result


# Parser

def parse_db_hybrid_search_result(res:SearchResult)->list[dict[str,str]]:
    result = {}
    for hits in res:
        for i,hit in enumerate(hits):
            result[i+1] = {
                "recipe_name":hit.get('name'),
                "ingredients":hit.get('ingredients'),
                "review":hit.get('synthetic_review')
            }
    return result