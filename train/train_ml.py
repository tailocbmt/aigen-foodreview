import json

from matplotlib import pyplot as plt
import shap
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, hamming_loss
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

    # ── MODALITY INDEPENDENT REPORTS ───────────────────────────────────────────
    if y_true.shape[1] >= 2:
        print(f'\n{"-"*30} Modality Reports {"-"*30}')

        # Column 0 = Text Modality
        print(f"\n[TEXT MODALITY] Classification Report on {dataset_name}:")
        print(classification_report(y_true[:, 0], y_pred[:, 0], labels=[0, 1], digits=4,
                                    target_names=['Generated (0)', 'Authentic (1)'], zero_division=0))

        # Column 1 = Image Modality
        print(f"\n[IMAGE MODALITY] Classification Report on {dataset_name}:")
        print(classification_report(y_true[:, 1], y_pred[:, 1], labels=[0, 1], digits=4,
                                    target_names=['Generated (0)', 'Authentic (1)'], zero_division=0))

    # Existing fallback per-label printout tracking
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


def load_data(path_train, path_val, path_test):
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
    clf = GridSearchCV(model, params, cv=5, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    print('Best estimator: ')
    print(clf.best_estimator_)
    if save:
        with open(f'./output/{model_name}_multilabel.pickle', 'wb') as f:
            pickle.dump(clf, f)
    return clf.best_estimator_


def predict(model, X_test):
    pred = model.predict(X_test)
    return pred


def generate_shap_plots(best_estimator, X_test, y_test, model_name):
    """
    Generates and saves SHAP summary plots for each label in a MultiOutput tree-based model.
    """
    print("\n--- Generating SHAP Feature Importance Plots ---")

    # 1. Ensure save directory exists
    save_dir = os.path.join("evons_data", "visual")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Extract target label names
    # Assumes y_test is a pandas DataFrame. If it's a numpy array, provide generic names.
    if hasattr(y_test, 'columns'):
        target_labels = y_test.columns.tolist()
    else:
        # Fallback if y_test lost its column names during transform_data
        target_labels = ['label_text', 'label_image']

    # 3. Verify the model structure supports MultiOutput
    if not hasattr(best_estimator, 'estimators_'):
        print("Warning: SHAP extraction requires a MultiOutputClassifier for multilabel tasks.")
        print("Skipping SHAP visualization.")
        return

    # 4. Generate a plot for each label
    for i, target_col in enumerate(target_labels):
        print(f"Generating SHAP plot for {target_col}...")
        try:
            # Extract the specific Random Forest/Tree trained for this exact label
            label_estimator = best_estimator.estimators_[i]

            # Initialize the SHAP explainer
            explainer = shap.TreeExplainer(label_estimator)
            shap_values = explainer.shap_values(X_test)

            # Extract values for Class 1 (Predicting 'Generated/Fake')
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values[:, :, 1] if len(
                    shap_values.shape) == 3 else shap_values

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.title(
                f"SHAP Feature Importance ({model_name}): {target_col.upper()}")

            # Generate summary plot
            shap.summary_plot(shap_values_to_plot, X_test, show=False)

            # Save and close
            save_path = os.path.join(
                save_dir, f"shap_importance_{model_name}_{target_col}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"Saved plot to '{save_path}'")

        except Exception as e:
            print(f"Could not generate SHAP plot for {target_col}. Error: {e}")
            plt.close()


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

    # ==========================================
    # INTEGRATED SHAP VISUALIZATION
    # ==========================================
    # Only run this if the selected model is tree-based (like Random Forest)
    if 'rf' in model_name.lower() or 'forest' in model_name.lower() or 'tree' in model_name.lower():
        generate_shap_plots(best_estimator, X_test, y_test, model_name)
    else:
        print(
            "\nNote: Skipping SHAP analysis. Selected model is not a tree-based ensemble.")


if __name__ == '__main__':
    # ./aigen_data/train_dataset.csv
    # ./aigen_data/val_dataset.csv
    # ./aigen_data/test_dataset.csv

    path_train = "./evons_data/train_multilabel.csv"
    path_val = "./evons_data/val_multilabel.csv"
    path_test = "./evons_data/test_multilabel.csv"
    model_name = args.model
    pipeline_sklearn(model_name, path_train, path_val, path_test)
