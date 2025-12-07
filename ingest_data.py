import pandas as pd
import pickle
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embeddings():
    logger.info("--- Starting Data Ingestion ---")
    
    # 2. Load the CSV
    try:
        df = pd.read_csv('shows.csv')
        logger.info(f"Loaded {len(df)} shows from shows.csv")
    except FileNotFoundError:
        logger.error("Error: shows.csv not found. Please make sure it is in the same folder.")
        return

    # Dictionary to store our data: { "Show Title": [Vector] }
    show_vector_dict = {}

    # 3. Loop through the shows and generate embeddings
    logger.info("Generating embeddings...")
    
    for index, row in df.iterrows():
        title = row['Title']
        description = row['Description']
        genres = row['Genres']
        
        # FEATURE ENGINEERING: Concatenating Genre and Description for better matching
        # We put Genre first to give it slightly more context weight at the beginning
        combined_text = f"Genres: {genres}. Description: {description}"
        
        try:
            # Call OpenAI API
            response = client.embeddings.create(
                input=combined_text,
                model="text-embedding-3-small" # Efficient and cheap model
            )
            
            # Extract the vector
            vector = response.data[0].embedding
            
            # Save to our dictionary
            show_vector_dict[title] = vector
            
            # Optional: Print progress every 10 shows
            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1}/{len(df)} shows...")
                
        except Exception as e:
            logger.error(f"Error processing {title}: {e}")

    # 4. Save to Pickle file
    with open('show_vectors.pkl', 'wb') as f:
        pickle.dump(show_vector_dict, f) #maps show titles to their embeddings
        
    logger.info("--- Success! ---")
    logger.info("Saved %d show vectors to 'show_vectors.pkl'", len(show_vector_dict))
    logger.info("You can now run the main program without calling the Embedding API again.")

if __name__ == "__main__":
    create_embeddings()