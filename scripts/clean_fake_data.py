import pandas as pd
import re

# Load CSV
df = pd.read_csv("data/evons_original_captions.csv")

# Function to clean text


def clean_text(text):
    if pd.isna(text):
        return text
    # Remove [Title] and [News paragraph]
    text = re.sub(r'\[(Title|News paragraph)\]', '', text)
    # Strip whitespace
    return text.strip()


# Apply to column
df["real_text"] = (
    df["title"].fillna("").astype(str).str.strip() + ". " +
    df["description"].fillna("").astype(str).str.strip()
)
df["fake_text"] = df["fake_description"].apply(clean_text)

# Save back
df.to_csv("real_evons_captions.csv", index=False)
