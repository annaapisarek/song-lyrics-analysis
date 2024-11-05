# Pop Songs Lyrics Analysis (1990-2024)

## Overview
A comprehensive analysis of Billboard chart-topping songs from 1990 to 2024, examining lyrical content, sentiment, and trends across different genres. This project combines Natural Language Processing (NLP) with Pandas and SQL to understand how song lyrics have evolved over time.

## Dataset Scale
- 34 years of music history
- 12,598 songs
- 13 distinct genres

## Data Sources

1. **Billboard Charts**
   - Kaggle dataset containing Billboard chart information

2. **Genius API**
   - Used for lyrics retrieval

3. **Spotify API**
   - Genre classification and additional metadata

## Methodology

### Data Processing Pipeline

#### 1. Merging 
- Implemented fuzzy search & regex for song-lyrics matching
- Combined data from multiple sources

#### 2. Cleaning
- Text tokenization and lemmatization
- Profanity identification
- Removal of non-essential information

#### 3. Enrichment
- Added metrics for lyrics length
- Calculated vocabulary richness
- Genre classification
- Additional feature engineering

### Natural Language Processing

* **Preprocessing**
  * NLTK toolkit for text processing

* **Sentiment Analysis**
  * VADER sentiment analyzer for short text analysis

* **Topic Modeling**
  * Latent Dirichlet Allocation (LDA) for theme identification
 
### Notable SQL Insights

#### Vocabulary Range
- **Richest**: Reggae (avg. 310 unique words)
- **Poorest**: "Around the World" by Daft Punk

#### Content Analysis
- 25%+ songs focus on Love and Loss themes
- ~85% of Hip Hop/Rap songs contain profanities

#### Sentiment Analysis
- **Most negative**: Hip Hop/Rap
- **Most positive**: Country

#### Most Featured Artist
- Taylor Swift (216 songs)

## Key Findings

### Hypothesis Testing Results

| Hypothesis | Outcome | Test Method |
|------------|---------|-------------|
| Genres are linked to specific topics | ✅ Supported | Chi2 contingency |
| More profanities in 21st century vs 90s | ✅ Supported | T-test |
| Hip Hop/Rap has richer vocabulary | ✅ Supported | T-test |
| Pop has more love songs than other genres | ✅ Supported | Two-Proportion Z-Test |



## Technical Challenges & Lessons Learned

### 1. LDA Model Training
* Complex parameter tuning required
* Multiple approaches needed for refinement

### 2. API Usage
* Genius API performance issues
* Implementation of retry mechanisms
* Batch processing for optimization

### 3. Resource Management
* Extensive library utilization
* Integration of multiple clients
* Academic paper references

## Tools & Technologies

### Analysis Tools
* Data Analysis: Pandas, SciPy
* Database: SQL
* NLP: NLTK, VADER
* Topic Modeling: LDA

### External Services
* Genius API
* Spotify API

## Author
**Anna Pisarek**
* Linguist with music industry experience

## More details
https://docs.google.com/presentation/d/1omwthfL1xnLFFi6E5GrkcP4E4LW-XUo63a_L5A8liBw/edit?usp=sharing

## Kanban board
https://trello.com/b/lQWTsXPE/lyrics-analysis-project

