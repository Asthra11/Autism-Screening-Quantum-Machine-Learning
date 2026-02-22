import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def run_qsvm(df: pd.DataFrame):
    """
    Quantum Support Vector Machine (QSVM)
    Uses last column as target automatically
    """

    # -----------------------------
    # Use last column as target
    # -----------------------------
    target = df.columns[-1]

    # Take first 30 rows and first 2 features
    X = df.drop(columns=[target]).iloc[:30, :2]
    y = df[target].iloc[:30]

    # Convert text labels to numeric if needed
    if y.dtype == "object":
        y = pd.factorize(y)[0]

    y = y.astype(int)

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Quantum Kernel
    # -----------------------------
    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    K_train = qkernel.evaluate(X_train)
    K_test = qkernel.evaluate(X_test, X_train)

    # -----------------------------
    # Classical SVM on Quantum Kernel
    # -----------------------------
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)

    preds = svm.predict(K_test)

    return accuracy_score(y_test, preds)