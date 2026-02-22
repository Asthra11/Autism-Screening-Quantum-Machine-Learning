import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data
from utils.data_utils import get_target_column


def run_logistic(user_input):
    """
    Logistic Regression for local execution
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

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    user_input = scaler.transform(user_input)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(user_input)[0]
    conf = np.max(model.predict_proba(user_input)) * 100
    acc = accuracy_score(y_test, model.predict(X_test))

    return pred, acc, conf
