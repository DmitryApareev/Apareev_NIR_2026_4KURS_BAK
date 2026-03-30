from sklearn.metrics import roc_auc_score, f1_score

def evaluate(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred)
    }