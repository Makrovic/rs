import pandas as pd

# Load the user interaction dataset
interactions = pd.read_csv("user_interaction_dataset.csv")

# User input for user ID
user_id = int(input("Enter the user ID: "))

# Filter interactions for the specified user ID
user_interactions = interactions[interactions["user_id"] == user_id]

# Create an empty user profile dictionary
user_profile = {}

# Update the user profile with interaction weights
for _, row in user_interactions.iterrows():
    laptop_id = row["laptop_id"]
    weight = row["weight"]
    user_profile[laptop_id] = weight


# Print the user profile
print(f"User Profile for User {user_id}:")
print(user_profile)
# for laptop_id, weight in user_profile.items():
#     print(f"- Laptop {laptop_id}: Weight = {weight}")
