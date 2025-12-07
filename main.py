import pickle
import sys
from thefuzz import process
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # TODO: genai phase

if __name__ == "__main__":
    main()