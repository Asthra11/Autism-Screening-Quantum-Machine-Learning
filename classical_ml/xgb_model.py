import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data
from utils.data_utils import get_target_column


def run_xgb(user_input):
    """
    XGBoost classifier for local execution
    Returns: prediction, accuracy, confidence
    """

    # Load dataset
    df = load_data()
    target_col = get_target_column(df)

    # Use only first 10 Q-CHAT features
    X = df.drop(columns=[target_col]).iloc[:, :10]
    y = df[target_col]

    # Encode target
    y = LabelEncoder().fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(user_input)[0]
    conf = np.max(model.predict_proba(user_input)) * 100
    acc = accuracy_score(y_test, model.predict(X_test))

    return pred, acc, conf
