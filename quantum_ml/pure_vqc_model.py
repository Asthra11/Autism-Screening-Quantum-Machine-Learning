import numpy as np
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.metrics import accuracy_score


def run_pure_vqc(X_train, X_test, y_train, y_test):

    # ✅ FIX: Convert Pandas Series → NumPy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_qubits = X_train.shape[1]

    feature_map = ZFeatureMap(
        feature_dimension=n_qubits,
        reps=1
    )

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2
    )

    optimizer = COBYLA(maxiter=40)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    print("\n⚛️ Training PURE Quantum VQC model...")
    vqc.fit(X_train, y_train)

    y_pred = vqc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc