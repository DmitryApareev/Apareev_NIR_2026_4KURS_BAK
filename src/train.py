from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        "LogReg": LogisticRegression(max_iter=2000),
        "RF": RandomForestClassifier(n_estimators=200),
        "GB": GradientBoostingClassifier()
    }