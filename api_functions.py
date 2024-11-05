### Import libraries
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import requests  
from requests.exceptions import HTTPError
import os 
import pandas as pd
import lyricsgenius
import fuzzywuzzy
import tqdm


### Genius API functions 

def setup_genius(api_key, timeout=10):
    """Initialize Genius API client with rate limiting and timeout"""
    genius = lyricsgenius.Genius(
        api_key,
        timeout=timeout,
        retries=3
    )
    genius.verbose = False
    genius.remove_section_headers = True
    return genius

def clean_lyrics(lyrics):
    """Clean up lyrics text by removing annotations and extra whitespace"""
    if not lyrics:
        return ""
    
    # Remove [Verse 1], [Chorus], etc.
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Remove empty lines and Genius annotations
    lyrics = re.sub(r'^\d+Contributors?$', '', lyrics, flags=re.MULTILINE)
    lyrics = re.sub(r'^\s*$\n', '', lyrics, flags=re.MULTILINE)
    # Remove extra whitespace
    lyrics = ' '.join(lyrics.split())
    return lyrics.strip()

def clean_song_title(title):
    """Clean special characters from song title for better matching"""
    if not title:
        return ""
    # Remove special quotes and normalize apostrophes
    title = title.replace('"', '').replace('"', '').replace("'", "")
    # Remove any special characters that might cause issues
    title = re.sub(r'[^\w\s\'-]', '', title)
    return title.strip()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def search_with_retry(genius, song, artist):
    """Search for a song with retry logic"""
    try:
        return genius.search_song(song, artist)
    except Timeout:
        print(f"\nTimeout occurred while searching for {song} by {artist}. Retrying...")
        raise
    except Exception as e:
        print(f"\nError occurred while searching for {song} by {artist}: {str(e)}")
        raise

def find_best_match(genius, artist, song, min_score=80):
    """
    Search for a song and return the best match including lyrics
    Returns (url, lyrics, match_score) or (None, None, 0) if no good match found
    """
    try:
        # Clean the song title before searching
        cleaned_song = clean_song_title(song)
        search_results = search_with_retry(genius, cleaned_song, artist)
        
        if not search_results:
            return None, None, 0
        
        title_score = fuzz.ratio(search_results.title.lower(), song.lower())
        artist_score = fuzz.ratio(search_results.artist.lower(), artist.lower())
        combined_score = (title_score * 0.6) + (artist_score * 0.4)
        
        if combined_score >= min_score:
            lyrics = clean_lyrics(search_results.lyrics)
            return search_results.url, lyrics, combined_score
        return None, None, combined_score
        
    except Exception as e:
        print(f"\nFailed to process {song} by {artist} after retries: {str(e)}")
        return None, None, 0

def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(checkpoint_file, processed_indices):
    """Save progress to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(list(processed_indices), f)

def add_genius_data(df, api_key, artist_col='Artist', song_col='Song', 
                    checkpoint_file='genius_checkpoint.json',
                    save_every=10):
    """
    Add Genius URLs and lyrics to the dataframe with progress bar and automatic saving
    """
    genius = setup_genius(api_key)
    
    # Initialize columns 
    for col in ['genius_url', 'lyrics', 'match_score']:
        if col not in df.columns:
            df[col] = None
    
    # Load checkpoint
    processed_indices = load_checkpoint(checkpoint_file)
    
    # Get unprocessed indices
    remaining_indices = [idx for idx in df.index if idx not in processed_indices]
    
    if not remaining_indices:
        print("All songs have been processed!")
        return df
    
    print(f"\nProcessing {len(remaining_indices)} songs...")
    
    # Process with progress bar
    pbar = tqdm(remaining_indices, desc="Fetching lyrics")
    failed_songs = []
    
    for i, idx in enumerate(pbar):
        row = df.loc[idx]
        song = row[song_col]
        artist = row[artist_col]
        
        # Update progress bar description
        pbar.set_description(f"Processing: {song[:30]}...")
        
        # Get lyrics and update dataframe
        url, lyrics, score = find_best_match(genius, artist, song)
        df.at[idx, 'genius_url'] = url
        df.at[idx, 'lyrics'] = lyrics
        df.at[idx, 'match_score'] = score
        
        # Track failed songs
        if lyrics is None:
            failed_songs.append((song, artist))
        
        # Mark as processed
        processed_indices.add(idx)
        
        # Save checkpoint and data periodically
        if (i + 1) % save_every == 0:
            save_checkpoint(checkpoint_file, processed_indices)
            save_results(df, "lyrics_data_intermediate.csv")
            
        time.sleep(1)  # Rate limiting between songs
    
    # Final save
    save_checkpoint(checkpoint_file, processed_indices)
    
    # Report failed songs
    if failed_songs:
        print("\nFailed to process the following songs:")
        for song, artist in failed_songs:
            print(f"- {song} by {artist}")
    
    return df

def save_results(df, output_path="lyrics_data.csv"):
    """Save the results with proper encoding for lyrics"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to {output_path}")


def clean_artist_name(artist):
    """Remove 'FEATURING' or 'FEAT.' and everything after it from artist name"""
    # List of possible featuring indicators
    featuring_terms = ['FEATURING', 'FEAT.', 'FEAT', 'FT.', 'FT', 'INTRODUCING', 'WITH', 'X']
    
    # Convert to uppercase for consistent comparison
    artist_upper = artist.upper()
    
    # Find the earliest occurrence of any featuring term
    min_index = len(artist)
    for term in featuring_terms:
        index = artist_upper.find(term)
        if index != -1 and index < min_index:
            min_index = index
    
    # Return the cleaned artist name, trimmed of whitespace
    return artist[:min_index].strip()

def update_missing_lyrics(csv_path, api_key):
    """Update CSV file with missing lyrics using cleaned artist names"""
    # Initialize Genius API
    genius = Genius(api_key)
    genius.verbose = False  # Turn off status messages
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Counter for tracking updates
    updates = 0
    
    # Iterate through rows where lyrics are missing
    for index, row in df.iterrows():
        if pd.isna(row['lyrics']):
            # Clean the artist name
            cleaned_artist = clean_artist_name(row['Artist'])
            
            try:
                # Search for the song
                song = genius.search_song(row['Song'], cleaned_artist)
                
                if song:
                    # Update the lyrics in the dataframe
                    df.at[index, 'lyrics'] = song.lyrics
                    updates += 1
                    print(f"Updated lyrics for {cleaned_artist} - {row['Song']}")
                else:
                    print(f"No lyrics found for {cleaned_artist} - {row['Song']}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {cleaned_artist} - {row['Song']}: {str(e)}")
    
    # Save the updated dataframe
    df.to_csv(csv_path, index=False)
    print(f"\nUpdated {updates} songs with missing lyrics")

def clean_lyrics(lyrics):
    """Clean lyrics by removing content before the word 'Lyrics' and unnecessary whitespace"""
    if pd.isna(lyrics):
        return lyrics
        
    # Find the position of 'Lyrics' (case insensitive)
    match = re.search(r'lyrics', lyrics, re.IGNORECASE)
    if match:
        # Get everything after the word 'Lyrics'
        lyrics = lyrics[match.end():].strip()
    
    # Remove extra whitespace and empty lines
    lyrics = '\n'.join(line.strip() for line in lyrics.split('\n') if line.strip())
    
    return lyrics

def clean_artist_name(artist):
    """Remove 'FEATURING' or 'FEAT.' and everything after it from artist name"""
    featuring_terms = ['FEATURING', 'FEAT.', 'FEAT', 'FT.', 'FT']
    artist_upper = artist.upper()
    
    min_index = len(artist)
    for term in featuring_terms:
        index = artist_upper.find(term)
        if index != -1 and index < min_index:
            min_index = index
    
    return artist[:min_index].strip()

def update_and_clean_lyrics(csv_path, api_key, batch_size=250):
    """Update missing lyrics and clean existing lyrics in batches"""
    # Initialize Genius API
    genius = Genius(api_key)
    genius.verbose = False
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Clean existing lyrics first
    print("Cleaning existing lyrics...")
    cleaned_count = 0
    for index, row in df.iterrows():
        if not pd.isna(row['lyrics']):
            cleaned_lyrics = clean_lyrics(row['lyrics'])
            if cleaned_lyrics != row['lyrics']:
                df.at[index, 'lyrics'] = cleaned_lyrics
                cleaned_count += 1
    
    print(f"Cleaned {cleaned_count} existing lyrics entries")
    
    # Save the cleaned data
    df.to_csv(csv_path, index=False)
    
    # Get indices of rows with missing lyrics
    missing_lyrics_indices = df[pd.isna(df['lyrics'])].index.tolist()
    total_missing = len(missing_lyrics_indices)
    
    if total_missing == 0:
        print("No missing lyrics found in the file.")
        return
    
    print(f"\nFound {total_missing} songs with missing lyrics")
    
    # Calculate number of batches
    num_batches = (total_missing + batch_size - 1) // batch_size
    
    # Initialize counters
    updates = 0
    errors = 0
    
    # Process missing lyrics in batches
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_missing)
        current_batch_indices = missing_lyrics_indices[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_num + 1}/{num_batches}")
        batch_updates = 0
        start_time = datetime.now()
        
        # Process each song in the current batch
        for index in current_batch_indices:
            row = df.loc[index]
            cleaned_artist = clean_artist_name(row['Artist'])
            
            try:
                # Search for the song
                song = genius.search_song(row['Song'], cleaned_artist)
                
                if song:
                    # Clean the lyrics before saving
                    cleaned_lyrics = clean_lyrics(song.lyrics)
                    df.at[index, 'lyrics'] = cleaned_lyrics
                    batch_updates += 1
                    updates += 1
                    print(f"✓ Updated: {cleaned_artist} - {row['Song']}")
                else:
                    errors += 1
                    print(f"✗ No lyrics found: {cleaned_artist} - {row['Song']}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                errors += 1
                print(f"✗ Error processing {cleaned_artist} - {row['Song']}: {str(e)}")
        
        # Save after each batch
        df.to_csv(csv_path, index=False)
        

### Spotifty API functions

load_dotenv()
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

import requests 

def get_access_token():
    """Function to get the access token for Spotify"""
    import requests 
    auth_response = requests.post(
        'https://accounts.spotify.com/api/token',
        data={'grant_type': 'client_credentials'},
        auth=('client_id', 'client_secret')
    )
    return auth_response.json().get('access_token')


def get_genre(artist_name):
    """Look up genres for each artist in Spotify"""
    access_token = get_access_token()
    headers = {'Authorization': f'Bearer {access_token}'}
    search_url = f'https://api.spotify.com/v1/search?q={artist_name}&type=artist'
    
    search_response = requests.get(search_url, headers=headers)
    search_results = search_response.json()
    
    if search_results['artists']['items']:
        return ', '.join(search_results['artists']['items'][0]['genres'])  
    return 'Unknown' 
        

def create_genre_clusters():
    """
    Creates a mapping dictionary for genre clustering.
    """
    genre_clusters = {
    # Hip Hop/Rap 
    '[hip hop]': 'Hip Hop/Rap',
    '[rap]': 'Hip Hop/Rap',
    '[atl hip hop]': 'Hip Hop/Rap',
    '[canadian hip hop]': 'Hip Hop/Rap',
    '[chicago rap]': 'Hip Hop/Rap',
    '[east coast hip hop]': 'Hip Hop/Rap',
    '[conscious hip hop]': 'Hip Hop/Rap',
    '[detroit hip hop]': 'Hip Hop/Rap',
    '[melodic rap]': 'Hip Hop/Rap',
    '[hip pop]': 'Hip Hop/Rap',
    '[dirty south rap]': 'Hip Hop/Rap',
    '[baton rouge rap]': 'Hip Hop/Rap',
    '[florida rap]': 'Hip Hop/Rap',
    '[dfw rap]': 'Hip Hop/Rap',
    '[pop rap]': 'Hip Hop/Rap',
    '[gangster rap]': 'Hip Hop/Rap',
    '[chicago drill]': 'Hip Hop/Rap',
    '[cali rap]': 'Hip Hop/Rap',
    '[memphis hip hop]': 'Hip Hop/Rap',
    '[emo rap]': 'Hip Hop/Rap',
    '[brooklyn drill]': 'Hip Hop/Rap',
    '[chicano rap]': 'Hip Hop/Rap',
    '[houston rap]': 'Hip Hop/Rap',
    '[bronx hip hop]': 'Hip Hop/Rap',
    '[indie pop rap]': 'Hip Hop/Rap',
    '[lgbtq+ hip hop]': 'Hip Hop/Rap',
    '[florida drill]': 'Hip Hop/Rap',
    '[alternative hip hop]': 'Hip Hop/Rap',
    '[battle rap]': 'Hip Hop/Rap',
    '[southern hip hop]': 'Hip Hop/Rap',
    '[hardcore hip hop]': 'Hip Hop/Rap',
    '[cloud rap]': 'Hip Hop/Rap',
    '[comedy rap]': 'Hip Hop/Rap',
    '[g funk]': 'Hip Hop/Rap',
    '[crunk]': 'Hip Hop/Rap',
    '[chopped and screwed]': 'Hip Hop/Rap',
    '[hyphy]': 'Hip Hop/Rap',
    '[wu fam]': 'Hip Hop/Rap',
    '[bronx drill]': 'Hip Hop/Rap',

    # Pop
    '[pop]': 'Pop',
    '[dance pop]': 'Pop',
    '[canadian pop]': 'Pop',
    '[art pop]': 'Pop',
    '[k-pop]': 'Pop',
    '[pop rock]': 'Pop',
    '[barbadian pop]': 'Pop',
    '[acoustic pop]': 'Pop',
    '[candy pop]': 'Pop',
    '[post-teen pop]': 'Pop',
    '[colombian pop]': 'Pop',
    '[electropop]': 'Pop',
    '[australian pop]': 'Pop',
    '[puerto rican pop]': 'Pop',
    '[singer-songwriter pop]': 'Pop',
    '[europop]': 'Pop',
    '[pop r&b]': 'Pop',
    '[alternative pop]': 'Pop',
    '[indie pop]': 'Pop',
    '[country pop]': 'Pop',
    '[pop soul]': 'Pop',
    '[bedroom pop]': 'Pop',
    '[boy band]': 'Pop',
    '[new wave pop]': 'Pop',
    '[adult standards]': 'Pop',
    '[girl group]': 'Pop',

    # R&B
    '[contemporary r&b]': 'R&B',
    '[r&b]': 'R&B',
    '[neo soul]': 'R&B',
    '[canadian contemporary r&b]': 'R&B',
    '[british soul]': 'R&B',
    '[alternative r&b]': 'R&B',
    '[quiet storm]': 'R&B',
    '[classic soul]': 'R&B',
    '[new jack swing]': 'R&B',
    '[motown]': 'R&B',
    '[bedroom soul]': 'R&B',
    '[post-disco soul]': 'R&B',
    '[retro soul]': 'R&B',
    '[new jack smooth]': 'R&B',
    '[souldies]': 'R&B',

    # Country
    '[contemporary country]': 'Country',
    '[country]': 'Country',
    '[classic oklahoma country]': 'Country',
    '[classic texas country]': 'Country',
    '[arkansas country]': 'Country',
    '[modern country pop]': 'Country',
    '[country road]': 'Country',
    '[canadian country]': 'Country',
    '[country dawn]': 'Country',
    '[country rock]': 'Country',
    '[alberta country]': 'Country',
    '[classic country pop]': 'Country',
    '[bakersfield sound]': 'Country',
    '[western americana]': 'Country',
    '[bluegrass]': 'Country',

    # Electronic/Dance
    '[edm]': 'Electronic/Dance',
    '[eurodance]': 'Electronic/Dance',
    '[diva house]': 'Electronic/Dance',
    '[dance rock]': 'Electronic/Dance',
    '[brostep]': 'Electronic/Dance',
    '[big room]': 'Electronic/Dance',
    '[complextro]': 'Electronic/Dance',
    '[big beat]': 'Electronic/Dance',
    '[classic house]': 'Electronic/Dance',
    '[disco house]': 'Electronic/Dance',
    '[electro]': 'Electronic/Dance',
    '[dutch edm]': 'Electronic/Dance',
    '[acid house]': 'Electronic/Dance',
    '[dance]': 'Electronic/Dance',
    '[house]': 'Electronic/Dance',
    '[australian dance]': 'Electronic/Dance',
    '[downtempo]': 'Electronic/Dance',
    '[bubblegum dance]': 'Electronic/Dance',
    '[bass house]': 'Electronic/Dance',
    '[dutch house]': 'Electronic/Dance',
    '[chicago house]': 'Electronic/Dance',
    '[filter house]': 'Electronic/Dance',
    '[bouncy house]': 'Electronic/Dance',
    '[chill house]': 'Electronic/Dance',
    '[detroit house]': 'Electronic/Dance',
    '[new beat]': 'Electronic/Dance',
    '[german techno]': 'Electronic/Dance',
    '[hardcore techno]': 'Electronic/Dance',
    '[new italo disco]': 'Electronic/Dance',
    '[euphoric hardstyle]': 'Electronic/Dance',
    '[classic progressive house]': 'Electronic/Dance',
    '[bmore]': 'Electronic/Dance',
    '[belgian dance]': 'Electronic/Dance',
    '[kids dance party]': 'Electronic/Dance',
    '[dream trance]': 'Electronic/Dance',

    # Rock
    '[alternative metal]': 'Rock',
    '[alternative rock]': 'Rock',
    '[modern rock]': 'Rock',
    '[glam metal]': 'Rock',
    '[permanent wave]': 'Rock',
    '[album rock]': 'Rock',
    '[hard rock]': 'Rock',
    '[irish rock]': 'Rock',
    '[dance rock]': 'Rock',
    '[glam rock]': 'Rock',
    '[classic rock]': 'Rock',
    '[rock drums]': 'Rock',
    '[modern folk rock]': 'Rock',
    '[australian rock]': 'Rock',
    '[garage rock]': 'Rock',
    '[soft rock]': 'Rock',
    '[grunge]': 'Rock',
    '[rock]': 'Rock',
    '[punk rock]': 'Rock',
    '[psychedelic rock]': 'Rock',
    '[folk rock]': 'Rock',
    '[progressive rock]': 'Rock',
    '[country rock]': 'Rock',
    '[heartland rock]': 'Rock',
    '[celtic rock]': 'Rock',
    '[funk metal]': 'Rock',
    '[dance-punk]': 'Rock',
    '[nu metal]': 'Rock',
    '[industrial]': 'Rock',
    '[cowpunk]': 'Rock',
    '[comedy rock]': 'Rock',
    '[j-division]': 'Rock',
    '[classic canadian rock]': 'Rock',
    '[canadian rock]': 'Rock',
    '[russian metal]': 'Rock',
    '[art rock]': 'Rock',
    '[beatlesque]': 'Rock',
    '[new wave]': 'Rock',
    '[british invasion]': 'Rock',
    '[rock-and-roll]': 'Rock',
    '[madchester]': 'Rock',
    '[glam punk]': 'Rock',
    '[deep new wave]': 'Rock',
    '[experimental guitar]': 'Rock',
    '[vocaloid metal]': 'Rock',
    '[comic metal]': 'Rock',
    '[metal]': 'Rock',

    # Latin
    '[reggaeton]': 'Latin',
    '[latin pop]': 'Latin',
    '[bachata]': 'Latin',
    '[corrido]': 'Latin',
    '[corridos tumbados]': 'Latin',
    '[latin hip hop]': 'Latin',
    '[latin arena pop]': 'Latin',
    '[pop reggaeton]': 'Latin',
    '[canadian latin]': 'Latin',
    '[tejano]': 'Latin',
    '[urbano mexicano]': 'Latin',
    '[urbano latino]': 'Latin',
    '[mambo chileno]': 'Latin',
    '[cubaton]': 'Latin',
    '[sertanejo]': 'Latin',
    '[dembow]': 'Latin',

    # Alternative/Indie
    '[neo mellow]': 'Alternative/Indie',
    '[indietronica]': 'Alternative/Indie',
    '[modern alternative rock]': 'Alternative/Indie',
    '[pov: indie]': 'Alternative/Indie',
    '[alaska indie]': 'Alternative/Indie',
    '[indie soul]': 'Alternative/Indie',
    '[modern indie pop]': 'Alternative/Indie',
    '[indie rock]': 'Alternative/Indie',
    '[emo]': 'Alternative/Indie',
    '[canadian indie]': 'Alternative/Indie',
    '[la indie]': 'Alternative/Indie',
    '[brooklyn indie]': 'Alternative/Indie',
    '[chicago indie]': 'Alternative/Indie',
    '[san marcos tx indie]': 'Alternative/Indie',
    '[nashville indie]': 'Alternative/Indie',
    '[boston indie]': 'Alternative/Indie',
    '[kentucky indie]': 'Alternative/Indie',
    '[el paso indie]': 'Alternative/Indie',
    '[new jersey indie]': 'Alternative/Indie',
    '[atlanta indie]': 'Alternative/Indie',
    '[eau claire indie]': 'Alternative/Indie',
    '[albuquerque indie]': 'Alternative/Indie',
    '[cologne indie]': 'Alternative/Indie',
    '[derby indie]': 'Alternative/Indie',
    '[bath indie]': 'Alternative/Indie',
    '[athens indie]': 'Alternative/Indie',
    '[alabama indie]': 'Alternative/Indie',

    # Jazz
    '[jazz]': 'Jazz',
    '[cool jazz]': 'Jazz',
    '[contemporary jazz]': 'Jazz',
    '[soul jazz]': 'Jazz',
    '[jazz funk]': 'Jazz',
    '[jazz trio]': 'Jazz',
    '[bebop]': 'Jazz',

    # Blues
    '[blues]': 'Blues',
    '[modern blues]': 'Blues',
    '[modern blues rock]': 'Blues',
    '[classic blues]': 'Blues',
    '[blues rock]': 'Blues',
    '[delta blues]': 'Blues',
    '[southern soul blues]': 'Blues',

    # Reggae
    '[reggae]': 'Reggae',
    '[reggae fusion]': 'Reggae',
    '[dancehall]': 'Reggae',
    '[dub]': 'Reggae',
    '[ska]': 'Reggae',
    '[roots reggae]': 'Reggae',
    '[lovers rock]': 'Reggae',

    # Gospel
    '[canadian ccm]': 'Gospel',
    '[praise]': 'Gospel',
    '[family gospel]': 'Gospel',
    '[christian a cappella]': 'Gospel',
    '[roots worship]': 'Gospel',

    # Folk
    '[american folk revival]': 'Folk',
    '[folk]': 'Folk',
    '[canadian celtic]': 'Folk',
    '[celtic]': 'Folk',
    '[irish singer-songwriter]': 'Folk',
    '[progressive bluegrass]': 'Folk',
    '[black americana]': 'Folk',

    # Other
    '[glee club]': 'Other',
    '[freestyle]': 'Other', 
    '[deep talent show]': 'Other',
    '[idol]': 'Other',
    '[alt z]': 'Other',
    '[lilith]': 'Other',
    '[funk]': 'Other',
    '[ectofolk]': 'Other',
    '[new romantic]': 'Other',
    '[hollywood]': 'Other',
    '[ccm]': 'Other',
    '[movie tunes]': 'Other',
    '[afrofuturism]': 'Other',
    '[mellow gold]': 'Other',
    '[hip house]': 'Other',
    '[piano rock]': 'Other',
    '[gospel]': 'Other',
    '[broadway]': 'Other',
    '[bounce]': 'Other',
    '[disco]': 'Other',
    '[miami bass]': 'Other',
    '[pluggnb]': 'Other',
    '[jam band]': 'Other',
    '[escape room]': 'Other',
    '[christian music]': 'Other',
    '[sad sierreno]': 'Other',
    '[atlanta bass]': 'Other',
    '[afrobeats]': 'Other',
    '[anime]': 'Other',
    '[other]': 'Other',
    '[comic]': 'Other',
    '[tropical]': 'Other',
    '[bboy]': 'Other',
    '[cartoon]': 'Other',
    '[talent show]': 'Other',
    '[hi-nrg]': 'Other',
    '[lo-fi vgm]': 'Other',
    '[minneapolis sound]': 'Other',
    '[novelty]': 'Other',
    '[jersey club]': 'Other',
    '[chamber ensemble]': 'Other',
    '[white noise]': 'Other',
    '[clean comedy]': 'Other',
    '[doo-wop]': 'Other',
    '[talentschau]': 'Other',
    '[orthodox chant]': 'Other',
    '[healing]': 'Other',
    '[new orleans funk]': 'Other',
    '[bossbeat]': 'Other',
    '[wu fam]': 'Other',
    '[pixel]': 'Other',
    '[swazi traditional]': 'Other',
    '[mezmur]': 'Other',
    '[432hz]': 'Other',
    '[electra]': 'Other',
    '[acoustic cover]': 'Other',
    '[chanson]': 'Other',
    '[ambeat]': 'Other',
    '[idol kayo]': 'Other'
    }
    
    return genre_clusters

def cluster_genres(df, genre_column, case_sensitive=False):
    """
    Clusters genres in a new column called Final genre.
    """
    # Create genre clusters mapping
    genre_clusters = create_genre_clusters()
    
    # Function to find matching genre
    def match_genre(genre):
        if not isinstance(genre, str):
            return 'Unknown'
            
        # Convert to lowercase if case-insensitive
        search_genre = genre if case_sensitive else genre.lower()
        
        # Try direct match
        if search_genre in genre_clusters:
            return genre_clusters[search_genre]
        
        # Try partial match (remove brackets for partial matching)
        search_genre_clean = search_genre.replace('[', '').replace(']', '')
        for key, value in genre_clusters.items():
            key_clean = key.replace('[', '').replace(']', '')
            if key_clean in search_genre_clean:
                return value
        
        return 'Other'
    
    # Apply clustering
    return df[genre_column].apply(match_genre)