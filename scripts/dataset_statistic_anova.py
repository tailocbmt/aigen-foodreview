import json
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# ==========================================
# 1. LOAD AND FLATTEN JSON DATA
# ==========================================
json_file_path = './evons_data/evons_metrics_output.json'

with open(json_file_path, 'r') as f:
    data = json.load(f)

rows = []
for item in data:
    item_id = item.get('id')

    # Extract Real Data
    real_data = item.get('real', {})
    for text_type in ['title', 'description', 'full_text']:
        if text_type in real_data:
            row = {'id': item_id, 'data_type': text_type,
                   'source': 'real', 'is_generated': 0}
            row.update(real_data[text_type])
            rows.append(row)

    if 'image' in real_data:
        row = {'id': item_id, 'data_type': 'image',
               'source': 'real', 'is_generated': 0}
        row.update(real_data['image'])
        rows.append(row)

    # Extract Generated Data
    gen_data = item.get('generated', {})
    for model_name, model_content in gen_data.items():
        # Text models
        for text_type in ['title', 'description', 'full_text']:
            if text_type in model_content:
                row = {'id': item_id, 'data_type': text_type,
                       'source': model_name, 'is_generated': 1}
                row.update(model_content[text_type])
                rows.append(row)
        # Image models
        if 'image' in model_content:
            row = {'id': item_id, 'data_type': 'image',
                   'source': model_name, 'is_generated': 1}
            row.update(model_content['image'])
            rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)


# ==========================================
# 2. DEFINITIONS & HELPER FUNCTIONS
# ==========================================
text_features = ['ARI', 'FR', 'DW', 'PPL', 'GFI', 'RT', 'WPS']
image_features = [
    'BRI', 'SAT', 'CON', 'CLA', 'WAR', 'COL',
    'SD', 'CD', 'TD', 'DD', 'ROT', 'HPVB', 'VPVB', 'HCVB', 'VCVB'
]


def format_mean_std(mean, std):
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.2f} ({std:.2f})"


def get_significance_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def generate_binary_summary(dataframe, features):
    """Calculates Real vs Generated (Overall)"""
    results = []
    for feature in features:
        authentic_data = dataframe[dataframe['is_generated']
                                   == 0][feature].dropna()
        generated_data = dataframe[dataframe['is_generated']
                                   == 1][feature].dropna()

        auth_mean, auth_std = authentic_data.mean(), authentic_data.std()
        gen_mean, gen_std = generated_data.mean(), generated_data.std()

        if len(authentic_data) > 0 and len(generated_data) > 0:
            f_stat, p_val = f_oneway(authentic_data, generated_data)
            stars = get_significance_stars(p_val)
            f_stat_str = f"{f_stat:.2f}{stars}"
        else:
            f_stat_str = "N/A"

        results.append({
            "Metric": feature,
            "Authentic (All)": format_mean_std(auth_mean, auth_std),
            "Generated (All)": format_mean_std(gen_mean, gen_std),
            "F-statistic": f_stat_str
        })
    return pd.DataFrame(results)


def generate_model_summary(dataframe, features, model_names):
    """Calculates Real vs specific models side-by-side"""
    results = []
    for feature in features:
        row_dict = {"Metric": feature}
        anova_groups = []

        # Get Real Baseline
        real_data = dataframe[dataframe['source'] == 'real'][feature].dropna()
        row_dict['Real'] = format_mean_std(real_data.mean(), real_data.std())
        if len(real_data) > 0:
            anova_groups.append(real_data)

        # Get individual models
        for model in model_names:
            model_data = dataframe[dataframe['source']
                                   == model][feature].dropna()
            row_dict[model.capitalize()] = format_mean_std(
                model_data.mean(), model_data.std())
            if len(model_data) > 0:
                anova_groups.append(model_data)

        # Calculate Multi-Group ANOVA
        if len(anova_groups) > 1:
            f_stat, p_val = f_oneway(*anova_groups)
            stars = get_significance_stars(p_val)
            row_dict['F-statistic (All Groups)'] = f"{f_stat:.2f}{stars}"
        else:
            row_dict['F-statistic (All Groups)'] = "N/A"

        results.append(row_dict)
    return pd.DataFrame(results)


# ==========================================
# 3. GENERATE AND PRINT TABLES
# ==========================================

print("==========================================")
print("       TEXT METRICS (REAL VS GEN)         ")
print("==========================================\n")

for t_type in ['title', 'description', 'full_text']:
    print(f"--- Summary for: {t_type.upper()} ---")
    subset_df = df[df['data_type'] == t_type]
    table = generate_binary_summary(subset_df, text_features)
    print(table.to_string(index=False))
    print("\n")


print("==========================================")
print("      TEXT METRICS (PER MODEL)            ")
print("==========================================\n")
# Assuming we want to compare models based on full_text.
# You can change 'full_text' to 'title' or 'description' if needed.
text_models = ['qwen', 'llama', 'mistral']
full_text_df = df[df['data_type'] == 'full_text']

model_text_table = generate_model_summary(
    full_text_df, text_features, text_models)
print("--- FULL TEXT comparison across models ---")
print(model_text_table.to_string(index=False))
print("\n")


print("==========================================")
print("       IMAGE METRICS (REAL VS GEN)        ")
print("==========================================\n")

image_df = df[df['data_type'] == 'image']
binary_img_table = generate_binary_summary(image_df, image_features)
print(binary_img_table.to_string(index=False))
print("\n")


print("==========================================")
print("       IMAGE METRICS (PER MODEL)          ")
print("==========================================\n")

image_models = ['sd', 'flux', 'z', 'sdxl']
model_img_table = generate_model_summary(
    image_df, image_features, image_models)
print(model_img_table.to_string(index=False))
print("\n")
