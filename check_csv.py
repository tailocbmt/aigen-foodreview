import pandas as pd

# Load the CSVs
df1 = pd.read_csv("~locnt/Downloads/dev1.csv")
df2 = pd.read_csv("~locnt/Downloads/original_captions_dev1.csv")

df1 = df1.drop(["llava_caption", "llava_generated_caption"], axis=1)
df1 = df1.loc[df1['saved_image_path'] != 'FAILED_GPU_ERROR']
df1 = df1.reset_index(drop=True)
df1["llava_caption"] = df2["llava_caption"]

df1.to_csv("dev1.csv", index=False)
