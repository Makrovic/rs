import pandas as pd
import numpy as np
import sklearn as sk
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack

# Load the dataset
df = pd.read_csv("laptop_sample.csv")

# Load the user interaction dataset
interactions = pd.read_csv("user_interaction_dataset.csv")

# define user id
user_id = 1

# Train Word2Vec model
sentences = df.iloc[:, 1:].apply(lambda x: x.tolist(), axis=1).tolist()
model = Word2Vec(sentences, min_count=1)


# Function to get the word embeddings for a given laptop
def get_laptop_embeddings(laptop_id):
    laptop_features = df[df["laptop_id"] == laptop_id].values.tolist()[0][1:]
    embeddings = []
    for feature in laptop_features:
        embeddings.extend(model.wv[feature])
    return embeddings


# Function to get user profile
def get_user_profile(user_id):
    interaction = interactions.loc[interactions["user_id"] == user_id]
    interacted_articles_profiles = []

    for _, row in interaction.iterrows():
        laptop_id = row["laptop_id"]
        article_index = df[df["laptop_id"] == laptop_id].index.values[0]
        embedding = get_laptop_embeddings(laptop_id)
        interacted_articles_profiles.append(embedding[article_index])

    interacted_articles_profiles = vstack(interacted_articles_profiles)

    # change array shape to vertical
    interactions_weight = np.array(interaction["weight"]).reshape(-1, 1)

    # axis 0 = sum in vertical axis
    interactions_weighted_avg = np.sum(
        interacted_articles_profiles.multiply(interactions_weight), axis=0
    )
    interactions_weighted_avg /= np.sum(interactions_weight)
    profile = np.asarray(interactions_weighted_avg).tolist()

    return profile


# Function to get the top 5 similar laptops
def get_top_similar_laptops(user_id):
    # Get the user profile
    user_profile = get_user_profile(user_id)

    # Calculate cosine similarity between user profile and all other laptops
    laptop_embeddings = []
    for index, row in df.iterrows():
        embeddings = get_laptop_embeddings(row["laptop_id"])
        laptop_embeddings.append(embeddings)

    similarity_scores = cosine_similarity(user_profile, laptop_embeddings)[0]

    # Sort the laptops based on similarity scores
    laptop_scores = list(enumerate(similarity_scores))
    laptop_scores = sorted(laptop_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 similar laptops
    top_similar_laptops = [
        (df.iloc[score[0]], score[1]) for score in laptop_scores[0:5]
    ]

    return top_similar_laptops


# User input for user ID
user_id = int(input("Enter the user ID: "))

# Find the top 5 similar laptops
similar_laptops = get_top_similar_laptops(user_id)

# Print the top 5 similar laptops with scores
print(f"Top 5 similar laptops to Laptop {user_id}:")
for laptop, score in similar_laptops:
    print(
        f"- Laptop {laptop['laptop_id']}: {laptop['brand']} {laptop['series']} (Score: {score})"
    )
