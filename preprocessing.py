import pandas as pd
import nltk
import gensim.downloader as api
import numpy as np
nltk.download('punkt')


# Gather raw data from the previously downloaded datasets
isis_data_raw = pd.read_csv('ISIS Twitter.csv')
twitter_data_raw = pd.read_csv('twitter_dataset.csv')


# Select only relevant columns (text and labels)
isis_data = isis_data_raw[['tweets', 'label']]
twitter_data = twitter_data_raw[['Text']]


# Reformat twitter data and add negative labels
twitter_data.rename(columns={'Text': 'tweets'}, inplace=True)
twitter_data['label'] = False


# Combine data into one dataset
combined_data = pd.concat([isis_data, twitter_data], ignore_index=True)


# Tokenize text
combined_data['tokens'] = combined_data['tweets'].apply(nltk.word_tokenize)

word2vec_model = api.load('word2vec-google-news-300')

# Function to get word2vec embedding for a token
def get_embedding(token):
    try:
        embedding = word2vec_model[token]
        return embedding
    except KeyError:
        # If token not in vocabulary, return 0 vector
        return np.zeros(len(word2vec_model['word']))

# Get word2vec embeddings for each token
combined_data['embeddings'] = combined_data['tokens'].apply(lambda tokens: [get_embedding(token) for token in tokens])


# Flatten the embeddings
combined_data['flattened_embeddings'] = combined_data['embeddings'].apply(lambda embeddings: [emb.flatten() if emb is not None else None for emb in embeddings])

# Concatenate flattened embeddings with labels
data_to_save = pd.concat([combined_data['flattened_embeddings'].apply(pd.Series), combined_data['label']], axis=1)

# Save to CSV
data_to_save.to_csv('final_data.csv', index=False)