from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def baseline_model(X, y):
    id_cols = [col for col in X.columns if str(col).startswith("SK_ID")]
    X = X.drop(columns=id_cols, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    lg = LogisticRegression(random_state=42, max_iter=1000)
    lg.fit(X_train_sc, y_train)

    y_pred_proba = lg.predict_proba(X_test_sc)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Baseline LR AUC: {baseline_auc:,.3f}")

    return lg, baseline_auc
