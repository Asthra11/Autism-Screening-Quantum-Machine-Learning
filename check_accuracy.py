from utils.data_loader import load_data
from quantum_ml.qsvm_model import run_qsvm

print("\nChecking QSVM Accuracy...")

X_train, X_test, y_train, y_test = load_data()
q_acc = run_qsvm(X_train, X_test, y_train, y_test)

print("Final QSVM Accuracy:", q_acc)