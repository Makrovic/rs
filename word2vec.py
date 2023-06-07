import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Sample laptop data
laptop_data = {
    "laptop_id": [1, 2, 3, 4],
    "company": ["HP", "Dell", "Lenovo", "Apple"],
    "product": ["HP Pavilion", "Dell XPS", "Lenovo ThinkPad", "MacBook Pro"],
    "inches": [15.6, 13.3, 14.0, 16.0],
    "weight": [4.2, 2.7, 3.9, 4.3],
    "cpu": ["Intel i5", "Intel i7", "AMD Ryzen 7", "Apple M1"],
    "ram": [8, 16, 16, 16],
    "storage": ["512GB SSD", "1TB SSD", "512GB SSD", "512GB SSD"],
    "screen": [True, True, False, True],
}

laptop_df = pd.DataFrame(laptop_data)

# Sample user interactions data
interaction_data = {
    "user_id": [1, 2, 3, 4, 5],
    "laptop_id": [1, 2, 3, 4, 1],
    "event": ["like", "wishlist", "cart", "wishlist", "like"],
    "weight": [1, 2, 3, 2, 1],
}

interaction_df = pd.DataFrame(interaction_data)

# Extract the user profile from interaction data
user_profile = interaction_df[interaction_df["user_id"] == 2]["event"].str.cat(sep=" ")

# Preprocess the laptop descriptions and user profile
laptop_sentences = [description.lower().split() for description in laptop_df["product"]]
user_sentence = user_profile.lower().split()

# Train Word2Vec model
model = Word2Vec(laptop_sentences, vector_size=100, min_count=1)

# Transform laptop descriptions into average vectors using Word2Vec
laptop_vectors = np.array(
    [
        np.mean(
            [model.wv[word] for word in sentence if word in model.wv]
            or [np.zeros(100)],
            axis=0,
        )
        for sentence in laptop_sentences
    ]
)

# Transform user profile into average vector using Word2Vec
user_vector = np.mean(
    [model.wv[word] for word in user_sentence if word in model.wv] or [np.zeros(100)],
    axis=0,
)

# Concatenate user profile vector with laptop vectors
vectors = np.concatenate((laptop_vectors, [user_vector]), axis=0)

# Calculate cosine similarities between user profile vector and laptop description vectors
cosine_similarities = cosine_similarity(vectors)

# Extract the similarity scores for the user profile
user_scores = cosine_similarities[-1, :-1]

# Combine cosine similarities with laptop IDs and weights
recommendation_scores = pd.DataFrame(
    {"laptop_id": laptop_df["laptop_id"], "score": user_scores}
)

# Merge interaction data with recommendation scores
recommendation_scores = pd.merge(
    recommendation_scores, interaction_df, on="laptop_id", how="left"
)

# Fill missing interactions with weights of 0
recommendation_scores["weight"].fillna(0, inplace=True)

# Calculate final recommendation scores by multiplying cosine similarities with interaction weights
recommendation_scores["final_score"] = (
    recommendation_scores["score"] * recommendation_scores["weight"]
)

# Sort laptops based on final recommendation scores
sorted_scores = recommendation_scores.sort_values(by="final_score", ascending=False)

# Print recommended laptops with attributes
print("Recommended Laptops:")
for idx, row in sorted_scores.iterrows():
    laptop_id = row["laptop_id"]
    laptop = laptop_df[laptop_df["laptop_id"] == laptop_id]
    print("Laptop ID:", laptop_id)
    print("Company:", laptop["company"].values[0])
    print("Product:", laptop["product"].values[0])
    print("Inches:", laptop["inches"].values[0])
    print("Weight:", laptop["weight"].values[0])
    print("CPU:", laptop["cpu"].values[0])
    print("RAM:", laptop["ram"].values[0])
    print("Storage:", laptop["storage"].values[0])
    print("Screen Size:", laptop["screen"].values[0])
    print("Final Score:", row["final_score"])
    print()
