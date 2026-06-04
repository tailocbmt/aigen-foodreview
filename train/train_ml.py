import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import argparse

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str,
                    help="Available models: 'lr', 'nb', 'rf', 'xgb' .")
parser.add_argument("--save", required=False, default=False,
                    type=bool, help="Default to false.")
args = parser.parse_args()


def load_data(path_train, path_val, path_test):
    train = pd.read_csv(path_train)
    val = pd.read_csv(path_val)
    test = pd.read_csv(path_test)
    return train, val, test


def transform_data(train, val, test):
    """Select pre-computed features from dataframe"""

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
        "vcvb": "VCVB"
    }
    # List of all features to use
    train = train.rename(columns=mapping)

    val = val.rename(columns=mapping)

    test = test.rename(columns=mapping)

    feature_columns = [
        # Text readability features
        'ARI',
        'DW',
        'FR',
        'GFI',
        'WPS',
        'RT',
        'PPL',

        # Image features
        'BRI',
        'SAT',
        'CLA',
        'WAR',
        'CON',
        'COL',
        'SD',
        'CD',
        'ID',
        'DD',
        'ROT',
        'HPVB',
        'VPVB',
        'HCVB',
        'VCVB'
    ]

    # Filter to only include columns that exist in the dataframes
    available_features = [
        col for col in feature_columns if col in train.columns]

    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"Warning: Missing features: {missing}")

    # Get labels
    y_train = train['label'] if 'label' in train.columns else train['is_fake']
    y_val = val['label'] if 'label' in val.columns else val['is_fake']
    y_test = test['label'] if 'label' in test.columns else test['is_fake']

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
    print(
        f"Val: {len(val) - len(X_val)} rows dropped, {len(X_val)} remaining")
    print(
        f"Test: {len(test) - len(X_test)} rows dropped, {len(X_test)} remaining")

    # Optional: Scale features for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Using {len(available_features)} features")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


models = {
    "lr": {'model': LogisticRegression(max_iter=2000), 'params': {"C": [0.001, 0.01, 0.1, 1, 10, 100]}},
    'nb': {'model': MultinomialNB(), 'params': {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}},
    'rf': {'model': RandomForestClassifier(), 'params': {"n_estimators": [75, 100, 125],
                                                         "max_depth": [12, 14, 16], "min_samples_split": [2, 3, 4]}},
    'xgb': {'model': XGBClassifier(), 'params': {"eta": [0.01, 0.03, 0.05], "max_depth": [4, 6, 8],
                                                 "n_estimators": [75, 100, 125]}}
}


def optimise(model_name: str, X_train, y_train, save):
    obj = models.get(model_name)
    model = obj.get('model')
    params = obj.get('params')
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    print('Best estimator: ')
    print(clf.best_estimator_)
    if save:
        with open(f'./output/{model_name}.pickle', 'wb') as f:
            pickle.dump(clf, f)
    return clf.best_estimator_


def metrics(pred, true):
    acc = round(accuracy_score(true, pred) * 100, 2)
    prec = round(precision_score(true, pred) * 100, 2)
    rec = round(recall_score(true, pred) * 100, 2)
    f1 = round(f1_score(true, pred) * 100, 2)
    print('Evaluation metrics on test set: ')
    print('Accuracy: ', acc, "%")
    print('Precision: ', prec, "%")
    print('Recall: ', rec, "%")
    print('F1-score', f1, "%")


def predict(model, X_test):
    pred = model.predict(X_test)
    return pred


def pipeline_sklearn(model_name, path_train="./data/train_dataset.csv", path_val="./data/val_dataset.csv", path_test="./data/test_dataset.csv", save=args.save):
    if model_name not in list(models.keys()):
        raise ValueError(
            f'Model invalid. Picke one among: {list(models.keys())}.')
    print(f'Model chosen: {model_name}.')
    print('Optimising..')
    train, val, test = load_data(path_train, path_val, path_test)
    X_train, y_train, X_val, y_val, X_test, y_test = transform_data(
        train, val, test)
    best_estimator = optimise(model_name, X_train, y_train, save)
    print(best_estimator)

    pred_val = predict(best_estimator, X_val)
    metrics(pred_val, y_val)

    pred = predict(best_estimator, X_test)
    metrics(pred, y_test)


if __name__ == '__main__':
    path_train = "./aigen_data/train_dataset.csv"
    path_val = "./aigen_data/val_dataset.csv"
    path_test = "./aigen_data/test_dataset.csv"
    model_name = args.model
    pipeline_sklearn(model_name, path_train, path_val, path_test)
