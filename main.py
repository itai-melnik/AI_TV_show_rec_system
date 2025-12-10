#region Imports
import pickle
import sys
import os
import requests
from dotenv import load_dotenv
from ingest_redis import logger
import redis
from redis.commands.search.query import Query
from thefuzz import process
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import logging
#endregion


logger = logging.getLogger(__name__)

load_dotenv()


gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


r = redis.Redis(host='localhost', port=6379, decode_responses=False)
INDEX_NAME = "shows_idx"


def get_all_titles_from_redis():
    """Fetch all keys starting with 'show:' to list"""   
    keys = r.keys("show:*")

    # decode bytes to strings
    titles = [k.decode('utf-8').replace("show:", "") for k in keys]
    return titles


def get_vector(title):
    """Retrieve a single vector from Redis."""
    #get the raw bytes and convert back to numpy
    data = r.hget(f"show:{title}", "embedding")
    if data:
        return np.frombuffer(data, dtype=np.float32)
    return None


def load_data():
    """Load the dictionary of show vectors from the pickle file."""
    try:
        with open('show_vectors.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Error: 'show_vectors.pkl' not found. Please run ingest.py first.")
        sys.exit(1)

def get_user_preferences(valid_titles):
    """
    Asks the user for shows, performs fuzzy matching, and confirms with the user.
    Returns a list of the exact titles found in our database.
    """
    while True:
      
        user_input = input("\nWhich TV shows did you really like watching? Separate them by a comma.\nMake sure to enter more than 1 show: ")
        
        # Split string by comma and remove whitespace
        raw_shows = [s.strip() for s in user_input.split(',') if s.strip()]

        if len(raw_shows) < 2:
            print("Please enter at least 2 shows.")
            continue

        # Fuzzy Matching
        matched_shows = []
        for raw_show in raw_shows:
            best_match, score = process.extractOne(raw_show, valid_titles)
            matched_shows.append(best_match)

        # Confirmation
        confirmation_str = ", ".join(matched_shows)
        print(f"\nMaking sure, do you mean {confirmation_str}? (y/n)")
        
        confirm = input().lower().strip()
        
        if confirm == 'y':
            return matched_shows
        else:
            print("\nSorry about that. Let's try again, please make sure to write the names of the tv shows correctly")



def search_redis(user_vector, k=5):
    """
    Perform the Vector Search (KNN) in Redis
    """
    #Prepare the query vector as bytes
    vec_bytes = user_vector.astype(np.float32).tobytes()

    # Construct the Query
    q = Query(f"*=>[KNN {k} @embedding $vec AS score]")\
        .sort_by("score")\
        .return_fields("title", "score")\
        .dialect(2) # Important: Vector search requires dialect 2 or greater

    params = {"vec": vec_bytes}
    
    # Execute
    results = r.ft(INDEX_NAME).search(q, query_params=params)
 
    return results.docs

#OLD FUNCTION 
#THIS FUNCTION IS NOT USED ANYMORE
def recommend_shows_loop(user_titles, embedding_data):
    """
    [DEPRECATED]
    1. Average the vectors of the user_titles.
    2. Loop through all_data to find similarities.
    3. Return top 5 matches (not including the shows the user already input).
    """
    print(f"Calculating recommendations based on: {user_titles}")

    # 1. Get vectors for the user's shows
    user_vectors = []
    for title in user_titles:
        user_vectors.append(embedding_data[title])
    
    # 2. Calculate the Average Vector
    profile_vector = np.mean(user_vectors, axis=0)
    
    # Reshape for sklearn
    profile_vector = profile_vector.reshape(1, -1)

    scores = []

    # 3. The Loop (Phase 1 Logic)
    for title, vector in embedding_data.items():
        # Skip the shows the user already input
        if title in user_titles:
            continue
        
        # Reshape current vector
        current_vector = np.array(vector).reshape(1, -1)
        
        # Calculate Similarity
        similarity = cosine_similarity(profile_vector, current_vector)[0][0]
        
        scores.append((title, similarity))

    # 4. Sort by similarity score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:5] # Return top 5




def generate_new_show_concept(base_shows, context_type="user's taste"):
    """
    Uses OpenAI to invent a new TV show concept based on a list of existing shows.
    Returns: (Title, Description)
    """
    print(f"\nThinking of a new show concept based on {context_type}...")
    
    prompt = f"""
    Create a concept for a new, original TV show based on the style and themes of these shows: {', '.join(base_shows)}.
    
    Return exactly two lines:
    Line 1: The Title of the new show
    Line 2: A short, 1-sentence description of the plot.
    Do not add bolding or extra text.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content.strip().split('\n')
    # Basic cleanup to ensure we get title and desc
    #TODO: add better structure to the response
    title = content[0].replace("Title:", "").strip()
    description = content[1].replace("Description:", "").strip() if len(content) > 1 else "A mysterious show."
    
    return title, description

def generate_show_poster(title, description):
    """
    Uses DALL-E 3 to generate a poster.
    """

    print(f"Painting the poster for '{title}' using DALL-E 3...")
    
    prompt = f"A high quality movie poster for a TV show named '{title}'. The show is about: {description}. Cinematic lighting, 4k."

    try:
        response = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

        image_url = response.data[0].url
            
        # Download the image from the URL
        img_data = requests.get(image_url).content
        image = Image.open(BytesIO(img_data))
        
        clean_title = "".join(x for x in title if x.isalnum())
        filename = f"posters/poster_{clean_title}_dalle.png"
        image.save(filename)
        print(f"   Success! Saved {filename}")
        image.show()

    except Exception as e:
        print(f"Error generating image: {e}")




def main():
    # Load from Redis
    logger.info("Connecting to Redis...")
    
    all_titles = get_all_titles_from_redis()
   
    # User Input
    chosen_shows = get_user_preferences(all_titles)
    
    # Vector Math
    print("\nCalculating User Taste Profile...")
    user_vectors = [get_vector(t) for t in chosen_shows]
    profile_vector = np.mean(user_vectors, axis=0)

    # Redis Search
    print("Searching Redis Index (Vector Similarity)...")
    search_results = search_redis(profile_vector, k=10)
    
    print("\nHere are the tv shows that I think you would love:")
    
    recommendations_titles = []
    count = 0
    
    for doc in search_results:
        title = doc.title
        #if it's one of the shows the user typed
        if title in chosen_shows:
            continue
            
        # Note: 0 distance = 100% match
        score = float(doc.score)
        percentage = int((1 - (score / 2)) * 100)
        
        print(f"{title} ({percentage}%)")
        recommendations_titles.append(title)
        
        count += 1
        if count >= 5: 
            break

    #GenAI Layer
    print("\n------------------------------------------------")
    s1_title, s1_desc = generate_new_show_concept(chosen_shows, "your favorites")
    print(f"\nShow #1 ({s1_title}): {s1_desc}")
    generate_show_poster(s1_title, s1_desc)
    
    s2_title, s2_desc = generate_new_show_concept(recommendations_titles[:3], "my recommendations")
    print(f"\nShow #2 ({s2_title}): {s2_desc}")
    generate_show_poster(s2_title, s2_desc)

if __name__ == "__main__":
    main()