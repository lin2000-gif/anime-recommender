import csv
from pathlib import Path

import chromadb

base_path = Path(__file__).parent

COLLABORATIVE_FILTERING_CSV_FILE_PATH = (base_path / "../data/anime-embeddings/collaborative_filtering_embeddings.csv").resolve()
CONTENT_BASED_CSV_FILE_PATH = (base_path / "../data/anime-embeddings/content_based_embeddings.csv").resolve()
ANIME_DATASET_CSV_FILE_PATH = (base_path / "../data/anime_dataset.csv").resolve()

ANIME_ID_KEY = "anime_id"
ANIME_METADATA_KEY = "Anime Metadata"
COLLABORATIVE_EMBEDDINGS_KEY = "Collaborative Embeddings"
CONTENT_EMBEDDINGS_KEY = "Content Embeddings"
ANIME_COLLECTION_NAME = "anime_collection"
CHROMADB_PERSISTENCE_PATH = "D:\cosmosdb"

CHROMA_MAX_BATCH_SIZE = 5000

RELATIVE_WEIGHT_OF_COLLABORATIVE_EMBEDDING = 2

def load_csv_into_dict(filepath: str):
    result = {}
    with open(filepath, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            anime_id_key = row[ANIME_ID_KEY]
            row.pop(ANIME_ID_KEY)
            result[anime_id_key] = row
    return result

def combine_csv_data(anime_data_dict, collaborative_embedding_dict, content_based_embedding_dict):
    result = {}
    for anime_id, anime_attrs in anime_data_dict.items():
        if anime_id not in collaborative_embedding_dict:
            continue
        item = {}
        item[ANIME_METADATA_KEY] = {
            "Name": str(anime_attrs["Name"]),
            "English_name": str(anime_attrs["English name"]),
            "Image_url": str(anime_attrs["Image URL"])
        }
        item[COLLABORATIVE_EMBEDDINGS_KEY] = [float(v) for k,v in collaborative_embedding_dict[anime_id].items()]
        item[CONTENT_EMBEDDINGS_KEY] = [float(v) for k,v in content_based_embedding_dict[anime_id].items()]

        result[anime_id] = item
    print(f"The merged dataset has {len(result)} animes.")
    return result

def get_weighted_merged_embeddings(collaborative_dict, content_dict):
    weighted_collaborative_dict = [x*RELATIVE_WEIGHT_OF_COLLABORATIVE_EMBEDDING for x in collaborative_dict]
    return weighted_collaborative_dict + content_dict

def ingest_data_into_collection(merged_dataset, collection: chromadb.Collection):
    embeddings_list = []
    metadata_list = []
    ids_list = []
    for anime_id, anime_attrs in merged_dataset.items():
        ids_list.append(anime_id)
        metadata_list.append(anime_attrs[ANIME_METADATA_KEY])
        embeddings_list.append(get_weighted_merged_embeddings(anime_attrs[COLLABORATIVE_EMBEDDINGS_KEY], anime_attrs[CONTENT_EMBEDDINGS_KEY]))
    for start_idx in range(0, len(merged_dataset), CHROMA_MAX_BATCH_SIZE):
        end_idx = min(start_idx + CHROMA_MAX_BATCH_SIZE, len(merged_dataset))
        collection.add(
            ids=ids_list[start_idx: end_idx],
            embeddings=embeddings_list[start_idx: end_idx],
            metadatas=metadata_list[start_idx: end_idx]
        )
    print("Data Ingestion Complete")

def main():
    anime_dict = load_csv_into_dict(ANIME_DATASET_CSV_FILE_PATH)
    collaborative_dict = load_csv_into_dict(COLLABORATIVE_FILTERING_CSV_FILE_PATH)
    contect_dict = load_csv_into_dict(CONTENT_BASED_CSV_FILE_PATH)

    combined_data = combine_csv_data(anime_dict, collaborative_dict, contect_dict)

    client = chromadb.PersistentClient(path=CHROMADB_PERSISTENCE_PATH)
    anime_collection = client.create_collection(name=ANIME_COLLECTION_NAME)
    ingest_data_into_collection(combined_data, anime_collection)

main()