import chromadb
import requests
import os
from constants import *
from models import Anime, UserAnimeRating
from dotenv import load_dotenv

from jikanpy import Jikan
from backoff import on_exception, expo
from ratelimit import limits, RateLimitException

NUMBER_OF_RESULTS = 10

load_dotenv()

class VectorDb():
    def __init__(self):
        self.client = chromadb.PersistentClient(CHROMADB_PERSISTENCE_PATH)
        self.collection = self.client.get_collection(ANIME_COLLECTION_NAME)

    def get_similar_animes(self, anime_id: str):
        embeddings = self.collection.get(ids=[anime_id], include = ["embeddings"])["embeddings"]
        response = self.collection.query(query_embeddings=embeddings,n_results=NUMBER_OF_RESULTS + 1)
        result = {}
        for idx, anime_id in enumerate(response['ids'][0]):
                metadata_item = response['metadatas'][0][idx]
                result[anime_id] = Anime(
                    mal_id=anime_id,
                    title=metadata_item[NAME_KEY],
                    english_title=metadata_item[ENGLISH_NAME_KEY],
                    image_url=metadata_item[IMAGE_URL_KEY],
                    popularity=metadata_item[POPULARITY_KEY],
                    favourites=metadata_item[FAVOURITE_KEY],
                    members=metadata_item[MEMBERS_KEY]
                )
        return result        


class JikanAdapter:
    def __init__(self):
        self.jikan = Jikan()
    
    @on_exception(expo, RateLimitException, max_tries=5)
    @limits(calls=1, period=1)
    def get_anime_details(self, anime_id):
        result = self.jikan.anime(id=anime_id, extension='full')["data"]
        return Anime(
            mal_id=str(result["mal_id"]),
            title=result["title"],
            english_title=result["title_english"],
            image_url=result["images"]["jpg"]["large_image_url"],
            popularity=result["popularity"],
            favourites=result["favorites"],
            members=result["members"]
        )
    
    def get_anime_recommendations(self, anime_id, topk=3):
        result = self.jikan.anime(id=anime_id, extension='recommendations')["data"]
        recommended_animes = []
        for recommendation in result[0: topk]:
            recommended_animes.append(self.get_anime_details(recommendation["entry"]["mal_id"]))
        return recommended_animes


class MalAdapter:
    def __init__(self):
        self.base_url = "https://api.myanimelist.net/v2/"
        self.client_id = os.getenv("MAL_CLIENT_ID")

        if not self.client_id:
            raise ValueError("Your MAL_CLIENT_ID is not set in the .env file")
        
        self.headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-MAL-CLIENT-ID': self.client_id
        }
    
    def call(self, uri, method="get", params=None, *args, **kwargs):
        requester = getattr(requests, method.lower())
        url = self.base_url + uri
        response = requester(url=url,
                             headers=self.headers,
                             params=params,
                             *args,
                             **kwargs)
        response_json = response.json()
        return response_json

    
    def  get_user_anime_list(self, username):
        uri = f"users/{username}/animelist"
        query_params = {
            "fields": "list_status",
            "limit": 100
        }
        result = self.call(uri=uri, method="get", params=query_params)["data"]
        user_anime_list = []
        for rating in result:
            user_anime_list.append(UserAnimeRating(
                mal_id=str(rating["node"]["id"]),
                score=int(rating["list_status"]["score"]),
                status=str(rating["list_status"]["status"])
            ))
        return user_anime_list
