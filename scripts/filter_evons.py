import pandas as pd

# Read the CSV file
df = pd.read_csv("data/evons.csv")

# Filter rows
filtered_df = df[(df["is_fake"] == 0) & (df["is_valid_image"] == 1)]

# Save the filtered data to a new CSV
filtered_df.to_csv("data/real_valid_img_evons.csv", index=False)

print(filtered_df.head())
print(f"Number of rows after filtering: {len(filtered_df)}")
