import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA


def run_vqc(df: pd.DataFrame):
    """
    Variational Quantum Classifier (VQC)
    Uses last column as target automatically
    """

    # -----------------------------
    # Use last column as target
    # -----------------------------
    target = df.columns[-1]

    # Take only first 40 rows and first 2 features
    X = df.drop(columns=[target]).iloc[:40, :2]
    y = df[target].iloc[:40]

    # Convert labels to numeric (if needed)
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
    # Quantum Circuit
    # -----------------------------
    n_qubits = 2

    inputs = ParameterVector("x", n_qubits)
    weights = ParameterVector("θ", n_qubits)

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        qc.ry(inputs[i], i)
        qc.ry(weights[i], i)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights
    )

    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=50)
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return accuracy_score(y_test, preds)