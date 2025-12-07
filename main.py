import pickle
import sys
import os
from dotenv import load_dotenv
from thefuzz import process
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

load_dotenv()


try:
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print("Warning: Gemini Client could not start (Check GEMINI_API_KEY or OPENAI_API_KEY). Image generation will be skipped.")
    gemini_client = None
    openai_client = None

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
        # 1. Ask the user for input
        user_input = input("\nWhich TV shows did you really like watching? Separate them by a comma.\nMake sure to enter more than 1 show: ")
        
        # Split string by comma and remove whitespace
        raw_shows = [s.strip() for s in user_input.split(',') if s.strip()]

        if len(raw_shows) < 2:
            print("Please enter at least 2 shows.")
            continue

        # 2. Fuzzy Matching
        matched_shows = []
        for raw_show in raw_shows:
            # process.extractOne returns a tuple: (Best Match String, Score)
            # We compare the user's input against the list of all valid titles
            best_match, score = process.extractOne(raw_show, valid_titles)
            matched_shows.append(best_match)

        # 3. Confirmation
        # Join the matched titles nicely for display
        confirmation_str = ", ".join(matched_shows)
        print(f"\nMaking sure, do you mean {confirmation_str}? (y/n)")
        
        confirm = input().lower().strip()
        
        if confirm == 'y':
            return matched_shows
        else:
            print("\nSorry about that. Let's try again, please make sure to write the names of the tv shows correctly")
            # The loop continues here, taking the user back to step #1


def recommend_shows(user_titles, embedding_data):
    """
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
    Uses Gemini 2.0 Flash to generate a poster via generate_content.
    """
    if not gemini_client:
        print("Skipping image generation (No Gemini Client).")
        return

    print(f"Painting the poster for '{title}' using Gemini 2.0 Flash...")
    
    prompt = f"A high quality movie poster for a TV show named '{title}'. The show is about: {description}. Cinematic lighting, 4k."
    
    #TODO: change model
    try:
        response = gemini_client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt],
    config=types.GenerateContentConfig(
        response_modalities=['Image']
    )
)

        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = part.as_image()
                # Save and Show
                clean_title = "".join(x for x in title if x.isalnum())
                filename = f"poster_{clean_title}.png"
                image.save(filename)
                print(f"Saved {filename}")
                image.show()

    except Exception as e:
        print(f"Error generating image: {e}")



def main():
    # Load the database (we only need the keys/titles for this step)
    print("Loading data...")
    show_data = load_data()
    all_titles = list(show_data.keys())
    
    # Get validated user input
    chosen_shows = get_user_preferences(all_titles)
    
    print("\nGreat! Generating recommendations now...")

    # Get Recommendations
    recommendations = recommend_shows(chosen_shows, show_data)
    
    print("\nHere are the tv shows that I think you would love:")
    for title, score in recommendations:
        percentage = int(score * 100)
        print(f"{title} ({percentage}%)")
    
    # 4. Generative AI Layer
    print("\n------------------------------------------------")
    print("I have also created just for you two shows which I think you would love.")
    
    # Show #1: Based on User Input
    s1_title, s1_desc = generate_new_show_concept(chosen_shows, "your favorites")
    print(f"\nShow #1 is based on the fact that you loved the input shows you gave me.")
    print(f"Its name is {s1_title} and it is about {s1_desc}.")
    generate_show_poster(s1_title, s1_desc)
    
    # Show #2: Based on Recommendations
    rec_titles = [r[0] for r in recommendations[:3]] # Take top 3 recommendations
    s2_title, s2_desc = generate_new_show_concept(rec_titles, "my recommendations")
    print(f"\nShow #2 is based on the shows that I recommended for you.")
    print(f"Its name is {s2_title} and it is about {s2_desc}.")
    generate_show_poster(s2_title, s2_desc)
    
    print("\nHere are also the 2 tv show ads. Hope you like them!")

if __name__ == "__main__":
    main()