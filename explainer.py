from typing import Dict, List
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import the centralized health suggestions
from health_suggestions import generate_personalized_suggestions


class ModelExplainer:
    def __init__(self, cardio_predictor, diabetes_predictor):
        self.cardio_predictor = cardio_predictor
        self.diabetes_predictor = diabetes_predictor
        self.cardio_lime = None
        self.diabetes_lime = None

    def setup_lime_explainers(self, cardio_X, diabetes_X):
        if not LIME_AVAILABLE:
            self.cardio_lime = None
            self.diabetes_lime = None
            return

        try:
            self.cardio_lime = lime.lime_tabular.LimeTabularExplainer(
                cardio_X.values, feature_names=list(cardio_X.columns),
                class_names=["No", "Yes"], discretize_continuous=True
            )
            self.diabetes_lime = lime.lime_tabular.LimeTabularExplainer(
                diabetes_X.values, feature_names=list(diabetes_X.columns),
                class_names=["No", "Yes"], discretize_continuous=True
            )
        except Exception:
            self.cardio_lime = None
            self.diabetes_lime = None

    def _format_explanation(self, weights: List[tuple]) -> Dict:
        top = weights[:10]
        return {"feature_importance": [{"feature": f, "importance": float(w)} for f, w in top]}

    def explain_diabetes_prediction(self, diabetes_data: Dict, method: str = "lime") -> Dict:
        if method == "lime" and self.diabetes_lime is not None and LIME_AVAILABLE:
            try:
                x = np.array([[diabetes_data[f] for f in self.diabetes_predictor.features]])
                predict_fn = lambda X: (
                    self.diabetes_predictor.model.predict_proba(X)
                    if hasattr(self.diabetes_predictor.model, "predict_proba")
                    else np.column_stack([1 - self.diabetes_predictor.model.predict(X),
                                         self.diabetes_predictor.model.predict(X)])
                )
                exp = self.diabetes_lime.explain_instance(x[0], predict_fn, num_features=10)
                return self._format_explanation(exp.as_list())
            except Exception:
                pass

        # Fallback explanation
        return self._fallback_explanation(self.diabetes_predictor)

    def explain_cardio_prediction(self, cardio_data: Dict, method: str = "lime") -> Dict:
        if method == "lime" and self.cardio_lime is not None and LIME_AVAILABLE:
            try:
                x = np.array([[cardio_data[f] for f in self.cardio_predictor.features]])
                predict_fn = lambda X: (
                    self.cardio_predictor.model.predict_proba(X)
                    if hasattr(self.cardio_predictor.model, "predict_proba")
                    else np.column_stack([1 - self.cardio_predictor.model.predict(X),
                                         self.cardio_predictor.model.predict(X)])
                )
                exp = self.cardio_lime.explain_instance(x[0], predict_fn, num_features=10)
                return self._format_explanation(exp.as_list())
            except Exception:
                pass

        # Fallback explanation
        return self._fallback_explanation(self.cardio_predictor)

    def _fallback_explanation(self, predictor) -> Dict:
        fi = []
        model = predictor.model
        feats = predictor.features
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            fi = sorted(zip(feats, vals), key=lambda t: abs(t[1]), reverse=True)[:10]
        elif hasattr(model, "coef_"):
            vals = model.coef_.ravel()
            fi = sorted(zip(feats, vals), key=lambda t: abs(t[1]), reverse=True)[:10]
        else:
            fi = [(f, 0.0) for f in feats[:10]]
        return self._format_explanation(fi)


class HealthSuggestionGenerator:
    """Generates health suggestions using the centralized health_suggestions.py module"""
    def generate_suggestions(self, patient_data: Dict, top_features: List[Dict]) -> List[Dict]:
        # Use generate_personalized_suggestions from health_suggestions.py
        suggestions = generate_personalized_suggestions(patient_data)

        # Optionally append top feature info from model explanations
        for feature in top_features:
            fname = feature.get("feature")
            importance = feature.get("importance", 0)
            suggestions.append({
                "feature": f"Important Feature: {fname}",
                "suggestion": f"This feature contributed significantly to the risk prediction (importance={importance:.2f})."
            })
        return suggestions


if __name__ == "__main__":
    pass
