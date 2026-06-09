import json

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from modules.utils import multilabel_accuracy
import argparse
import ast

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str,
                    help="Available models: 'lr', 'nb', 'rf', 'xgb' .")
parser.add_argument("--save", required=False, default=False,
                    type=bool, help="Default to false.")
args = parser.parse_args()


def multilabel_metrics(y_true, y_pred, dataset_name="Test"):
    """Calculate comprehensive metrics for multi-label classification"""

    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Calculate all metrics
    acc = accuracy_score(y_true, y_pred)
    multi_acc = multilabel_accuracy(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)

    # Print results
    print(f'\n{"="*50}')
    print(f'Evaluation metrics on {dataset_name} set:')
    print(f'{"="*50}')
    print(f'Accuracy (standard): {acc:.4f} ({acc*100:.2f}%)')
    print(
        f'Multilabel Accuracy (exact match): {multi_acc:.4f} ({multi_acc*100:.2f}%)')
    print(f'Macro Precision: {prec:.4f} ({prec*100:.2f}%)')
    print(f'Macro Recall: {rec:.4f} ({rec*100:.2f}%)')
    print(f'Macro F1-Score: {macro_f1:.4f} ({macro_f1*100:.2f}%)')
    print(f'Micro F1-Score: {micro_f1:.4f} ({micro_f1*100:.2f}%)')
    print(f'Samples F1-Score: {samples_f1:.4f} ({samples_f1*100:.2f}%)')
    print(f'Hamming Loss: {hamming:.4f} ({hamming*100:.2f}%)')

    # Per-label metrics (optional but helpful)
    print(f'\n--- Per-label Metrics ---')
    n_labels = y_true.shape[1]
    for i in range(n_labels):
        label_acc = accuracy_score(y_true[:, i], y_pred[:, i])
        label_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        print(f'Label {i}: Acc={label_acc:.4f}, F1={label_f1:.4f}')

    # Return metrics as dictionary
    return {
        'accuracy': acc,
        'multilabel_accuracy': multi_acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'samples_f1': samples_f1,
        'hamming_loss': hamming
    }


def json_to_custom_csv_pandas(json_file_path, csv_file_path):
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create rows as per your structure
    rows = []
    for row_id, row in enumerate(data):
        row_metrics = {
            "id": row_id,
            "is_fake": row.get('is_fake', ''),
            "real_title": row.get('real_title_m', ''),
            "real_description": row.get('real_des_m', ''),
            "real_full_text": row.get('combined_real_m', ''),
            "real_image": row.get('real_img_m', ''),
        }

        for model, model_data in row_metrics["generated"].iterrows():
            row_metrics[f"{model}_generated_title"] = model_data["title"]
            row_metrics[f"{model}_generated_description"] = model_data["description"]
            row_metrics[f"{model}_generated_full_text"] = model_data["full_text"]

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(csv_file_path, index=False, encoding='utf-8')

    print(f"Converted {len(df)} rows to {csv_file_path}")
    return df


def load_data(path_train, path_val, path_test):
    if path_train.endswith('.json'):
        df = json_to_custom_csv_pandas(path_train)

    train = pd.read_csv(path_train)
    val = pd.read_csv(path_val)
    test = pd.read_csv(path_test)

    # Convert label strings to lists if needed (e.g., "[0,1]" -> [0,1])
    for df in [train, val, test]:
        if 'label' in df.columns and isinstance(df['label'].iloc[0], str):
            df['label'] = df['label'].apply(ast.literal_eval)

    return train, val, test


def transform_data(train, val, test):
    """Select pre-computed features from dataframe for multi-label classification"""

    mapping = {
        "automated_readability_index": "ARI",
        "difficult_words": "DW",
        "flesch_reading_ease": "FR",
        "gunning_fog": "GFI",
        "words_per_sentence": "WPS",
        "reading_time": "RT",
        "ppl": "PPL",
        "bright": "BRI",
        "sat": "SAT",
        "clar": "CLA",
        "cont": "CON",
        "warm": "WAR",
        "colorf": "COL",
        "sd": "SD",
        "cd": "CD",
        "td": "TD",
        "diag_dom": "DD",
        "rot": "ROT",
        "hpvb": "HPVB",
        "vpvb": "VPVB",
        "hcvb": "HCVB",
        "vcvb": "VCVB",
        "VPVP": "VPVB",  # Fix typo
    }

    # Rename columns
    train = train.rename(columns=mapping)
    val = val.rename(columns=mapping)
    test = test.rename(columns=mapping)

    feature_columns = [
        # Text readability features
        'ARI', 'DW', 'FR', 'GFI', 'WPS', 'RT', 'PPL',

        # Image features
        'BRI', 'SAT', 'CLA', 'WAR', 'CON', 'COL', 'SD', 'CD', 'TD',
        'DD', 'ROT', 'HPVB', 'VPVB', 'HCVB', 'VCVB'
    ]

    # Filter to only include columns that exist in the dataframes
    available_features = [
        col for col in feature_columns if col in train.columns]

    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"Warning: Missing features: {missing}")

    # Get labels (multi-label)
    if 'label' in train.columns:
        y_train_raw = train['label']
        y_val_raw = val['label']
        y_test_raw = test['label']
    else:
        y_train_raw = train['is_fake']
        y_val_raw = val['is_fake']
        y_test_raw = test['is_fake']

    # Convert labels to 2D array if they're not already
    if isinstance(y_train_raw.iloc[0], (list, np.ndarray)):
        y_train = np.array(y_train_raw.tolist())
        y_val = np.array(y_val_raw.tolist())
        y_test = np.array(y_test_raw.tolist())
    else:
        # If labels are binary single-column, convert to 2D
        y_train = np.array(y_train_raw).reshape(-1, 1)
        y_val = np.array(y_val_raw).reshape(-1, 1)
        y_test = np.array(y_test_raw).reshape(-1, 1)

    # Extract features and drop rows with NaN
    X_train = train[available_features].dropna()
    y_train = y_train[X_train.index]  # Align labels with dropped rows

    X_val = val[available_features].dropna()
    y_val = y_val[X_val.index]

    X_test = test[available_features].dropna()
    y_test = y_test[X_test.index]

    # Print info about dropped rows
    print(
        f"Train: {len(train) - len(X_train)} rows dropped, {len(X_train)} remaining")
    print(f"Val: {len(val) - len(X_val)} rows dropped, {len(X_val)} remaining")
    print(
        f"Test: {len(test) - len(X_test)} rows dropped, {len(X_test)} remaining")
    print(f"Label shape: {y_train.shape[1]} labels per sample")

    # Scale features for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Using {len(available_features)} features")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# Modified models for multi-label classification
models = {
    "lr": {'model': MultiOutputClassifier(LogisticRegression(max_iter=2000)),
           'params': {"estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]}},
    'nb': {'model': MultiOutputClassifier(MultinomialNB()),
           'params': {"estimator__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}},
    'rf': {'model': MultiOutputClassifier(RandomForestClassifier()),
           'params': {"estimator__n_estimators": [75, 100, 125],
                      "estimator__max_depth": [12, 14, 16],
                      "estimator__min_samples_split": [2, 3, 4]}},
    'xgb': {'model': MultiOutputClassifier(XGBClassifier()),
            'params': {"estimator__eta": [0.01, 0.03, 0.05],
                       "estimator__max_depth": [4, 6, 8],
                       "estimator__n_estimators": [75, 100, 125]}}
}


def optimise(model_name: str, X_train, y_train, save):
    obj = models.get(model_name)
    model = obj.get('model')
    params = obj.get('params')
    # Use micro F1 for multi-label
    clf = GridSearchCV(model, params, cv=5, scoring='f1_micro')
    clf.fit(X_train, y_train)
    print('Best estimator: ')
    print(clf.best_estimator_)
    if save:
        with open(f'./output/{model_name}_multilabel.pickle', 'wb') as f:
            pickle.dump(clf, f)
    return clf.best_estimator_


def multilabel_metrics(pred, true, dataset_name="Test"):
    """Calculate metrics for multi-label classification"""

    # Convert to numpy arrays if needed
    pred = np.array(pred)
    true = np.array(true)

    # Ensure 2D arrays
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.ndim == 1:
        true = true.reshape(-1, 1)

    n_labels = pred.shape[1]

    print(f'\n{"="*50}')
    print(f'Evaluation metrics on {dataset_name} set:')
    print(f'{"="*50}')

    # Per-label metrics
    print(f'\n--- Per-label Metrics ---')
    for i in range(n_labels):
        acc = round(accuracy_score(true[:, i], pred[:, i]) * 100, 2)
        prec = round(precision_score(
            true[:, i], pred[:, i], zero_division=0) * 100, 2)
        rec = round(recall_score(
            true[:, i], pred[:, i], zero_division=0) * 100, 2)
        f1 = round(f1_score(true[:, i], pred[:, i], zero_division=0) * 100, 2)
        print(f'Label {i}: Acc={acc}%, P={prec}%, R={rec}%, F1={f1}%')

    # Overall metrics
    print(f'\n--- Overall Metrics ---')

    # Exact match (all labels correct)
    exact_match = np.mean(np.all(pred == true, axis=1)) * 100
    print(f'Exact Match Ratio: {round(exact_match, 2)}%')

    # Hamming loss (percentage of wrong labels)
    hamming_loss = np.mean(pred != true) * 100
    print(f'Hamming Loss: {round(hamming_loss, 2)}%')

    # Micro-averaged metrics (aggregate all labels)
    micro_prec = precision_score(
        true, pred, average='micro', zero_division=0) * 100
    micro_rec = recall_score(
        true, pred, average='micro', zero_division=0) * 100
    micro_f1 = f1_score(true, pred, average='micro', zero_division=0) * 100
    print(
        f'Micro Avg - P={round(micro_prec, 2)}%, R={round(micro_rec, 2)}%, F1={round(micro_f1, 2)}%')

    # Macro-averaged metrics (average across labels)
    macro_prec = precision_score(
        true, pred, average='macro', zero_division=0) * 100
    macro_rec = recall_score(
        true, pred, average='macro', zero_division=0) * 100
    macro_f1 = f1_score(true, pred, average='macro', zero_division=0) * 100
    print(
        f'Macro Avg - P={round(macro_prec, 2)}%, R={round(macro_rec, 2)}%, F1={round(macro_f1, 2)}%')

    return {
        'exact_match': exact_match,
        'hamming_loss': hamming_loss,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }


def predict(model, X_test):
    pred = model.predict(X_test)
    return pred


def pipeline_sklearn(
    model_name,
    path_train="./data/train_dataset.csv",
    path_val="./data/val_dataset.csv",
    path_test="./data/test_dataset.csv",
    save=args.save
):
    if model_name not in list(models.keys()):
        raise ValueError(
            f'Model invalid. Pick one among: {list(models.keys())}.')

    print(f'Model chosen: {model_name} (Multi-label mode)')
    print('Loading and transforming data...')
    train, val, test = load_data(path_train, path_val, path_test)
    X_train, y_train, X_val, y_val, X_test, y_test = transform_data(
        train, val, test)

    print(f'\nOptimising hyperparameters...')
    best_estimator = optimise(model_name, X_train, y_train, save)
    print(f'\nBest estimator: {best_estimator}')

    print(f'\n--- Validation Set ---')
    pred_val = predict(best_estimator, X_val)
    multilabel_metrics(y_val, pred_val, "Validation")

    print(f'\n--- Test Set ---')
    pred = predict(best_estimator, X_test)
    multilabel_metrics(y_test, pred, "Test")


if __name__ == '__main__':
    # ./aigen_data/train_dataset.csv
    # ./aigen_data/val_dataset.csv
    # ./aigen_data/test_dataset.csv

    path_train = "./evons_data/train_multilabel.csv"
    path_val = "./evons_data/val_multilabel.csv"
    path_test = "./evons_data/test_multilabel.csv"
    model_name = args.model
    pipeline_sklearn(model_name, path_train, path_val, path_test)
