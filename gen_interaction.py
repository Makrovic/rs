import pandas as pd
import random

# Load the laptop dataset
laptops = pd.read_csv("laptop_sample.csv")

# Generate user interaction dataset
user_ids = [1, 2, 3, 4, 5]  # User IDs
laptop_ids = laptops["laptop_id"].tolist()  # Laptop IDs
weights = [1, 2, 3]  # Interaction weights: 1 for like, 2 for wishlist, 3 for cart

# Create an empty list to store the user interactions
interactions = []

# Generate interactions for each user and laptop
for user_id in user_ids:
    # Select a random number of interactions for the user
    num_interactions = random.randint(1, 5)

    # Randomly select laptops and assign weights for each interaction
    for _ in range(num_interactions):
        laptop_id = random.choice(laptop_ids)
        weight = random.choice(weights)
        interactions.append([user_id, laptop_id, weight])

# Create a DataFrame from the interactions list
interaction_data = pd.DataFrame(
    interactions, columns=["user_id", "laptop_id", "weight"]
)

# Save the user interaction dataset to a CSV file
interaction_data.to_csv("user_interaction_dataset.csv", index=False)
