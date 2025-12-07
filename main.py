import pickle
import sys
from thefuzz import process

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

def main():
    # Load the database (we only need the keys/titles for this step)
    print("Loading data...")
    show_data = load_data()
    all_titles = list(show_data.keys())
    
    # Get validated user input
    chosen_shows = get_user_preferences(all_titles)
    
    print("\nGreat! Generating recommendations now...")
    print(f"(Internal Debug) Validated shows to process: {chosen_shows}")
    
    # TODO: Implement the recommendation logic here

if __name__ == "__main__":
    main()