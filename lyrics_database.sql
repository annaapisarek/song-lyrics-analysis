-- Switch to the lyrics_analysis schema
USE lyrics_analysis;

-- Create the artists table
CREATE TABLE artists (
    artist_id INT AUTO_INCREMENT PRIMARY KEY,
    artist VARCHAR(255) NOT NULL
);

-- Create the genres table
CREATE TABLE genres (
    genre_id INT AUTO_INCREMENT PRIMARY KEY,
    genre VARCHAR(100) NOT NULL
);

-- Create the topics table
CREATE TABLE topics (
    topic_id INT AUTO_INCREMENT PRIMARY KEY,
    topic VARCHAR(100) NOT NULL
);

-- Create the songs table with foreign keys linking to artists, genres, and topics
CREATE TABLE songs (
    song_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    profanity BOOLEAN,
    length_without_stopwords INT,
    unique_words INT,
    word_frequency TEXT,
    sentiment_score FLOAT,
    sentiment VARCHAR(50),
    genre_id INT,
    topic_id INT,
    artist_id INT,
    
    -- Foreign key constraints
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id),
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
    FOREIGN KEY (artist_id) REFERENCES artists(artist_id)
);
-- Import genres.csv to genres table
LOAD DATA LOCAL INFILE '/Users/annapisarek/Desktop/Ironhack/FinalProject/genre_table.csv'
INTO TABLE genres
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Import artists.csv to artists table
LOAD DATA LOCAL INFILE '/Users/annapisarek/Desktop/Ironhack/FinalProject/artist_table.csv.csv'
INTO TABLE artists
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Import topics.csv to topics table
LOAD DATA LOCAL INFILE '/Users/annapisarek/Desktop/Ironhack/FinalProject/topic_table.csv'
INTO TABLE topics
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Import songs.csv to songs table
LOAD DATA LOCAL INFILE '/Users/annapisarek/Desktop/Ironhack/FinalProject/song_table.csv'
INTO TABLE songs
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Find the most popular artists

SELECT s.artist_id, COUNT(s.song_id), a.artist FROM songs s
JOIN artists AS a on
s.artist_id = a.artist_id
GROUP BY artist_id
ORDER BY COUNT(song_id) DESC
LIMIT 10;

-- Find the genre with most profanities

SELECT 
    g.genre,
    COUNT(s.song_id) AS total_songs,
    SUM(CASE WHEN s.profanity = 1 THEN 1 ELSE 0 END) AS profane_count,
    (SUM(CASE WHEN s.profanity = 1 THEN 1 ELSE 0 END) / COUNT(s.song_id)) * 100 AS profanity_share
FROM 
    songs s
JOIN 
    genres g ON g.genre_id = s.genre_id
GROUP BY 
    g.genre
ORDER BY 
    profanity_share DESC;
    
 
-- Find the song with the shortest lyrics

SELECT title, a.artist, g.genre, unique_words FROM songs s
JOIN genres g on
g.genre_id = s.genre_id
JOIN artists a on
a.artist_id = s.artist_id
WHERE unique_words > 0
ORDER BY unique_words ASC;   


-- Find the richest word count genre

SELECT  g.genre, AVG(unique_words) FROM songs s
JOIN genres g on
g.genre_id = s.genre_id
WHERE unique_words > 0
GROUP BY genre
ORDER BY AVG(unique_words) DESC;


-- Find most common topics

SELECT  t.topic, COUNT(song_id) FROM songs s
JOIN topics t on
t.topic_id = s.topic_id
GROUP BY topic
ORDER BY COUNT(song_id) DESC;

-- Positive sentiment share

SELECT 
    g.genre, 
    SUM(CASE WHEN s.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_count,
    (SUM(CASE WHEN s.sentiment = 'positive' THEN 1 ELSE 0 END) / COUNT(s.song_id)) * 100 AS positive_share 
FROM 
    songs s
JOIN 
    genres g ON g.genre_id = s.genre_id
GROUP BY 
    g.genre
ORDER BY 
    g.genre;