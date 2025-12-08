import pickle
import numpy as np
import redis
import logging
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

logger = logging.getLogger(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

INDEX_NAME = "shows_idx"
VECTOR_DIM = 1536 # OpenAI embedding size

def create_index():
    try:
        # Check if index exists
        r.ft(INDEX_NAME).info()
        logger.info("Index already exists.")
    except:
        # Define Schema
        schema = (
            TextField("title"),
            VectorField("embedding",
                "HNSW", { # HNSW is the fast vector search algorithm
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )
        
      
        r.ft(INDEX_NAME).create_index(
            schema, 
            definition=IndexDefinition(prefix=["show:"], index_type=IndexType.HASH)
        )
        logger.info("Created new index.")

def migrate_data():
    logger.info("Loading data from pickle...")
    try:
        with open('show_vectors.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error("Pickle file not found. Run ingest.py first.")
        return

    logger.info(f"Migrating {len(data)} shows to Redis...")
    
    pipeline = r.pipeline()
    for title, vector in data.items():
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        pipeline.hset(f"show:{title}", mapping={
            "title": title,
            "embedding": vector_bytes
        })
        
    pipeline.execute()
    logger.info("Migration complete!")

if __name__ == "__main__":
    create_index()
    migrate_data()