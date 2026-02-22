def run_svm(user_input):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    from utils.data_loader import load_data
    from utils.data_utils import get_target_column

    # Load full dataset
    df = load_data()

    # Detect target column
    target_col = get_target_column(df)

    # 🔹 SELECT ONLY Q-CHAT QUESTION COLUMNS (FIRST 10)
    X = df.drop(columns=[target_col]).iloc[:, :10]
    y = df[target_col]

    # Encode target
    y = LabelEncoder().fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (10 features only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    user_input = scaler.transform(user_input)

    # Train SVM
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(user_input)[0]
    conf = np.max(model.predict_proba(user_input)) * 100
    acc = accuracy_score(y_test, model.predict(X_test))

    return pred, acc, conf
