
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess_lyrics(df, text_column):
    """
    Preprocess lyrics to remove stopwords, tokenize, lemmatize, and count words. Also perform part-of-speech (POS) tagging.
    """
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    # Initialize preprocessor tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Preprocess lyrics
    text_with_stopwords = []
    text_without_stopwords = []
    original_length = []
    processed_length_with_stopwords = []
    processed_length_without_stopwords = []
    unique_words = []
    word_frequency = []
    pos_tags = []

    for lyrics in df[text_column]:
        # Check if the lyrics value is a string
        if isinstance(lyrics, str):
            # Basic cleaning
            text = lyrics.lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(\s*\)', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^a-z\'\s]', '', text)

            # Tokenization
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            tokens_without_stopwords = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            # Perform POS tagging
            pos_tagged_tokens = nltk.pos_tag(tokens_without_stopwords)

            # Calculate metadata
            text_with_stopwords.append(text)
            text_without_stopwords.append(' '.join(tokens_without_stopwords))
            original_length.append(len(tokens))
            processed_length_with_stopwords.append(len(tokens))
            processed_length_without_stopwords.append(len(tokens_without_stopwords))
            unique_words.append(len(set(tokens_without_stopwords)))
            word_frequency.append(str(dict(Counter(tokens_without_stopwords))))
            pos_tags.append(', '.join([f"{token}/{tag}" for token, tag in pos_tagged_tokens]))
        else:
            # Handle non-string values
            text_with_stopwords.append('')
            text_without_stopwords.append('')
            original_length.append(0)
            processed_length_with_stopwords.append(0)
            processed_length_without_stopwords.append(0)
            unique_words.append(0)
            word_frequency.append('')
            pos_tags.append('')

    # Add new columns to the input DataFrame
    df['Text_with_stopwords'] = text_with_stopwords
    df['Text_without_stopwords'] = text_without_stopwords
    df['Original_length'] = original_length
    df['Processed_length_without_stopwords'] = processed_length_without_stopwords
    df['Unique_words'] = unique_words
    df['Word_frequency'] = word_frequency
    df['Pos_tags'] = pos_tags

    return df

def analyze_sentiment(text_column, pos_tags):
    """
    Analyses the sentiment of song lyrics using VADER
    """
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Split the POS tags into individual token-tag pairs
    pos_tag_pairs = pos_tags.split(', ')
    
    sentiment_score = 0
    for pair in pos_tag_pairs:
        try:
            token, tag = pair.split('/')
            if tag.startswith('J'):  # Adjective
                sentiment_score += analyzer.polarity_scores(token)['compound'] * 1.2
            elif tag.startswith('R'):  # Adverb
                sentiment_score += analyzer.polarity_scores(token)['compound'] * 1.1
            elif tag.startswith('V'):  # Verb
                sentiment_score += analyzer.polarity_scores(token)['compound'] * 1.0
            else:
                sentiment_score += analyzer.polarity_scores(token)['compound'] * 0.8
        except ValueError:
            pass
    return sentiment_score


def detect_swear_words(df, text_column):
    """
    Checks for swear words and adds a boolean column to the DataFrame marking songs that contain profanities.
    """
    # Define basic swear words
    swear_words = ['fuck', 'shit', 'damn', 'bitch', 'ass', 'fucking', 'nigger', 'nigga', 'cunt', 'dick']
    
    # Escape special characters and create pattern
    escaped_words = [re.escape(word) for word in swear_words]
    pattern = r'\b(' + '|'.join(escaped_words) + r')\b'
    
    # Create the new column initialized to False
    df['has_swear'] = False
    
    # Process in chunks for large datasets
    chunk_size = 10000
    
    # Process the DataFrame in chunks
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]
        
        # Vectorized operation on the chunk
        mask = chunk[text_column].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
        
        df.iloc[start:end, df.columns.get_loc('has_swear')] = mask
    
    return df


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")).union({
    # Lyrics-specific stop words
    'mind', 'eye', 'head', 'turn', 'chorus', 'verse', 'bridge', 'repeat', 'instrumental', 'money', 'una', 'como', 'los', 'quiero', 'para', 'sin',
    'intro', 'outro', 'hook', 'pre', 'post', 'til', 'yeah', 'oh', 'ah', 'woah', 'ooh', 'uh', 'mm', 
    'ayy', 'nt', 's', 'm', 'na', 'nothing', 'much', 'going', 'done', 'like', 'bout', 'bad', 'huh',
    'around', 'ever', 'long', 'well', "'s", "'m", 'need', 'look', 'think', "'ll", 
    'see', 'little', 'fuck', 'would', 'say', 'want', 'back', 'shawty', 'black', 'right', 'girl', "'d", 'kanye', 'travis',
    'every', 'day', 'feel', 'tonight', 'give', 'night', 'ft', 'feat', 'thing', 'hey', 
    'come', 'never', 'good', 'way', 'man', 'also', 'woo', 'boy', 'something', 'whoa', 'blacks', 'kendrick', 'scott', 'john',
    'take', 'ima', 'lil', 'get', 'got', 'know', 'one', 'baby', 'let', 'make', 'said', 
    'put', 'shit', 'tell', 'really', 'time', 'still', 'heart', 'away', 'hand', 'even', 
    'ohoh', 'world', 'keep', 'cause', 'that', 'made', 'call', 'dont', 'ohohohohohohoh', 
    'oohoohooh', 'dohdohdoh', 'yall', 'young', 'big', 'upon', 'best', 'everybody', 
    'somebody', 'another', 'round', 'new', 'friend', 'without', 'first', 'everything', 'damn',
    'two', 'three', 'wish', 'watch', 'better', 'walk', 'four', 'people', 'room', 'though', 'many',
     'play', 'nah', 'bring', 'nigga', 'might', 'gon', 'nananana', 'fuckin', 
    'ohohoh', "fuckin'", 'lalalalala', 'remix', 'wan', 'wanna', 'could', 'low', 'uhhuh', 'babe',
    'liveget', 'que', 'turn', 'head', 'try', 'getting', 'left', 'came', 'always', 'goin', 'gettin',
    'okay', 'mean', 'real', 'eye', 'mind', 'money', 'bit', 'try', 'mean', 'left', 'may', 'took', 'hold',
    'real', 'hard', 'pretty', 'white', 'alright', 'song', 'music', 'beat', 'bang', 'move', 'heads', 'eyes', 'york', 'wayne','doin','bitch','men'
})

def preprocess_text(text):
     """
     Clean and tokenize text into a list of words.
     """
     if not isinstance(text, str):
        return []
    
     # Convert to lowercase
     text = text.lower()
    
     # Remove punctuation and numbers
     text = re.sub(r"[^\w\s]", "", text)
     text = re.sub(r"\d+", "", text)
    
     # Split into words
     words = text.split()
    
     # Remove stop words and lemmatize
     cleaned_words = [
     lemmatizer.lemmatize(word) 
     for word in words 
         if word not in stop_words and len(word) > 2
    ]
    
     return cleaned_words

from gensim.models.callbacks import PerplexityMetric
from gensim.models import LdaMulticore

def train_lda_model(df, num_topics):
    """
    Train LDA model on the text data and return results.
    """
    # Preprocess all texts
    df['Lyrics_tokenized'] = df['Text_without_stopwords'].apply(preprocess_text)
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(df['Lyrics_tokenized'])
    
    # Update the dictionary filtering
    dictionary.filter_extremes(no_below=300, no_above=0.3, keep_n=100000)

    # Re-create corpus with the filtered dictionary
    corpus = [dictionary.doc2bow(text) for text in df['Lyrics_tokenized']]

    # Train the LDA model with adjusted parameters
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        workers=10,
        passes=50,
        iterations=500,
        random_state=42,
        alpha=0.1,
        eta=0.1,
        decay=0.7,
        offset=64,
        minimum_probability=0.02,
        per_word_topics=True,
    )
    
    return lda_model, dictionary, corpus

def get_topic_words(lda_model, num_words=15):
    """
    Get the top words for each topic with their weights.
    """
    topics_df = pd.DataFrame(columns=['Topic', 'Word', 'Weight'])
    
    # Get topics with their word distributions
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, num_words)
        
        # Add each word and its weight to the DataFrame
        for word, weight in topic_words:
            topics_df = topics_df._append({
                'Topic': topic_id,
                'Word': word,
                'Weight': weight
            }, ignore_index=True)
    
    return topics_df

def print_topics(topics_df):
    """
    Print topics in a readable format.
    """
    print("\n=== Topic Keywords ===")
    
    for topic_id in range(topics_df['Topic'].max() + 1):
        # Get words for this topic
        topic_words = topics_df[topics_df['Topic'] == topic_id]
        
        # Create the topic string
        topic_str = " + ".join([
            f"{row['Weight']:.3f}*\"{row['Word']}\""
            for _, row in topic_words.iterrows()
        ])
        
        print(f"\nTopic {topic_id}:")
        print(topic_str)

def assign_topics_to_lyrics(lda_model, corpus, df):
    """
    Assign the most probable topic to each document.
    """
    # Get the dominant topic for each document
    lyrics_topics = []
    for doc_bow in corpus:
        # Get topic distribution for this document
        topic_probs = lda_model.get_document_topics(doc_bow)
        # Find the topic with highest probability
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        lyrics_topics.append(dominant_topic)
    
    return lyrics_topics

def analyze_lyrics(df, num_topics=13):
    """
    Main function to analyze lyrics and assign topics.
    """
    lda_model, dictionary, corpus = train_lda_model(df, num_topics)
    
    topics_df = get_topic_words(lda_model)
    
    df['Topic'] = assign_topics_to_lyrics(lda_model, corpus, df)
    
    # Print topic information
    print_topics(topics_df)
    
    # Print some basic statistics
    topic_counts = Counter(df['Topic'])
    for topic_id, count in sorted(topic_counts.items()):
        percentage = (count / len(df)) * 100
        print(f"Topic {topic_id}: {count} songs ({percentage:.1f}%)")
    
    return df, topics_df


def label_topics(topic_num):
    """
    Translate topics to topic labels.
    """
    labels = {
        0: "Street Life & Hustle",
        1: "Love and Loss",
        2: "Everyday Life",
        3: "Longing & Memories",
        4: "Family & Home",
        5: "Rebellion & Toughness",
        6: "Life Experience & Growth",
        7: "Flirt & Romance",
        8: "Freedom & Adventure",
        9: "Thrill-seeking",
        10: "Party",
        11: "Other Relationships & Feelings",
        12: "Introspection & Life Lessons"
      
     
       
  
    }
    return labels.get(topic_num, "Unknown")  
