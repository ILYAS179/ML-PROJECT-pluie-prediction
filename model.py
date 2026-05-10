from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def build_model():
    """Crée et retourne le modèle configuré."""
    model = LogisticRegression(
        class_weight='balanced',  # gère le déséquilibre 78/22
        max_iter=1000,            # assez d'itérations pour converger
        random_state=42
    )
    return model

def train_model(model, X_train, y_train):
    """Entraîne le modèle et le retourne."""
    model.fit(X_train, y_train)
    print("Modèle entraîné sur", X_train.shape[0], "exemples.")
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et affiche les métriques."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 45)
    print("      RAPPORT DE CLASSIFICATION")
    print("=" * 45)
    print(classification_report(y_test, y_pred,
                                target_names=['No Rain', 'Rain']))

    print("AUC-ROC :", round(roc_auc_score(y_test, y_proba), 3))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Rain', 'Rain'])
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    return y_pred, y_proba