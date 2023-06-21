import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("laptop_sample.csv")

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


# Function to get the top 5 similar laptops
def get_top_similar_laptops(laptop_id):
    # Get the embeddings for the target laptop
    target_embeddings = get_laptop_embeddings(laptop_id)

    # Calculate cosine similarity between target laptop and all other laptops
    laptop_embeddings = []
    for index, row in df.iterrows():
        embeddings = get_laptop_embeddings(row["laptop_id"])
        laptop_embeddings.append(embeddings)

    similarity_scores = cosine_similarity([target_embeddings], laptop_embeddings)[0]

    # Sort the laptops based on similarity scores
    laptop_scores = list(enumerate(similarity_scores))
    laptop_scores = sorted(laptop_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 similar laptops (excluding the same laptop)
    top_similar_laptops = [
        (df.iloc[score[0]], score[1]) for score in laptop_scores[1:6]
    ]

    return top_similar_laptops


# User input for laptop ID
laptop_id = int(input("Enter the laptop ID: "))

# Find the top 5 similar laptops
similar_laptops = get_top_similar_laptops(laptop_id)

# Print the top 5 similar laptops with scores
print(f"Top 5 similar laptops to Laptop {laptop_id}:")
for laptop, score in similar_laptops:
    print(
        f"- Laptop {laptop['laptop_id']}: {laptop['brand']} {laptop['series']} (Score: {score})"
    )
