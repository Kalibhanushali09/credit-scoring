import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def explain_model(model, X_train, X_test, max_samples=500):
    sample = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False)
    plt.close()
    return shap_values