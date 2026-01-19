import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def get_model_descriptions():
    """Return descriptions of available models"""
    return {
        'Random Forest': 'An ensemble method that combines multiple decision trees. Good for handling mixed data types and provides feature importance.',
        'Gradient Boosting': 'Sequential ensemble method that builds models iteratively. Often provides high accuracy but can be prone to overfitting.',
        'Logistic Regression': 'Linear model for binary classification. Fast, interpretable, and works well with linearly separable data.',
        'Support Vector Machine': 'Finds optimal decision boundary by maximizing margin. Effective for high-dimensional data.',
        'K-Nearest Neighbors': 'Non-parametric method that classifies based on similarity to k nearest neighbors. Simple but can be sensitive to irrelevant features.',
        'Decision Tree': 'Tree-based model that makes decisions through a series of questions. Highly interpretable but prone to overfitting.',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem. Fast and works well with small datasets.'
    }

def build_model_candidates():
    """Build dictionary of model candidates"""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }

class CardiovascularPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        
    def preprocess_data(self, df):
        """Preprocess cardiovascular data"""
        # Drop patient ID if exists
        if 'patientid' in df.columns:
            df = df.drop('patientid', axis=1)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        self.features = X.columns.tolist()
        
        return X, y
    
    def train(self, df, model_choice='Random Forest'):
        """Train cardiovascular model with improved accuracy"""
        X, y = self.preprocess_data(df)
        
        # Use 70-30 split for better training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all models and rank them
        models = build_model_candidates()
        results = []
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = 0.0
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC': auc
            })
        
        # Sort by accuracy (descending)
        results.sort(key=lambda x: x['Accuracy'], reverse=True)
        
        # Select the chosen model
        self.model = models[model_choice]
        self.model.fit(X_train_scaled, y_train)
        
        return {
            'ranked_models': results,
            'best_model': model_choice,
            'test_accuracy': accuracy_score(y_test, self.model.predict(X_test_scaled))
        }
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Ensure correct column order
        df = df.reindex(columns=self.features, fill_value=0)
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0, 1] if hasattr(self.model, 'predict_proba') else 0.5
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        
    def preprocess_data(self, df):
        """Preprocess diabetes data"""
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        self.features = X.columns.tolist()
        
        return X, y
    
    def train(self, df, model_choice='Random Forest'):
        """Train diabetes model with improved accuracy and 70-30 split"""
        X, y = self.preprocess_data(df)
        
        # Use 70-30 split as requested
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all models and rank them
        models = build_model_candidates()
        results = []
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = 0.0
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC': auc
            })
        
        # Sort by accuracy (descending)
        results.sort(key=lambda x: x['Accuracy'], reverse=True)
        
        # Select the chosen model
        self.model = models[model_choice]
        self.model.fit(X_train_scaled, y_train)
        
        return {
            'ranked_models': results,
            'best_model': model_choice,
            'test_accuracy': accuracy_score(y_test, self.model.predict(X_test_scaled))
        }
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Ensure correct column order
        df = df.reindex(columns=self.features, fill_value=0)
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0, 1] if hasattr(self.model, 'predict_proba') else 0.5
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']

class CombinedRiskPredictor:
    def __init__(self):
        self.cardio_predictor = None
        self.diabetes_predictor = None
    
    def predict_for_diabetes_patient(self, patient_data):
        """Make combined predictions for a diabetes patient"""
        # Separate data for each model
        cardio_data = {
            'age': patient_data.get('age', 40),
            'gender': patient_data.get('gender', 1),
            'chestpain': patient_data.get('chestpain', 0),
            'restingBP': patient_data.get('restingBP', 120),
            'serumcholestrol': patient_data.get('serumcholestrol', 200),
            'fastingbloodsugar': patient_data.get('fastingbloodsugar', 0),
            'restingrelectro': patient_data.get('restingrelectro', 0),
            'maxheartrate': patient_data.get('maxheartrate', 150),
            'exerciseangia': patient_data.get('exerciseangia', 0),
            'oldpeak': patient_data.get('oldpeak', 0.0),
            'slope': patient_data.get('slope', 0),
            'noofmajorvessels': patient_data.get('noofmajorvessels', 0),
        }
        
        diabetes_data = {
            'Pregnancies': patient_data.get('Pregnancies', 0),
            'Glucose': patient_data.get('Glucose', 100),
            'BloodPressure': patient_data.get('BloodPressure', 80),
            'SkinThickness': patient_data.get('SkinThickness', 20),
            'Insulin': patient_data.get('Insulin', 100),
            'BMI': patient_data.get('BMI', 25.0),
            'DiabetesPedigreeFunction': patient_data.get('DiabetesPedigreeFunction', 0.5),
            'Age': patient_data.get('Age', patient_data.get('age', 40)),
        }
        
        # Make predictions
        cardio_pred, cardio_prob = self.cardio_predictor.predict(cardio_data)
        diabetes_pred, diabetes_prob = self.diabetes_predictor.predict(diabetes_data)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            diabetes_pred, diabetes_prob, cardio_pred, cardio_prob,
            patient_data.get('age', 40), patient_data.get('BMI', 25.0)
        )
        
        return {
            'diabetes_prediction': diabetes_pred,
            'diabetes_probability': diabetes_prob,
            'cardiovascular_prediction': cardio_pred,
            'cardiovascular_probability': cardio_prob,
            'risk_assessment': risk_assessment
        }
    
    def _calculate_risk_assessment(self, diabetes_pred, diabetes_prob, cardio_pred, cardio_prob, age, bmi):
        """Calculate overall risk assessment based on probability ranges"""
        
        # Use actual probabilities from models (not inverted for negative predictions)
        diabetes_risk = diabetes_prob  # Always use the raw probability
        cardio_risk = cardio_prob      # Always use the raw probability
        
        # Age factor (normalized - higher age = higher risk)
        age_factor = min((age - 20) / 60.0, 1.0) if age > 20 else 0.0
        age_factor = max(age_factor, 0.0)  # Ensure non-negative
        
        # BMI factor (more balanced)
        if bmi < 18.5:
            bmi_factor = 0.1  # Underweight - slight risk
        elif bmi < 25:
            bmi_factor = 0.0  # Normal - no additional risk
        elif bmi < 30:
            bmi_factor = 0.1  # Overweight - slight risk
        else:
            bmi_factor = 0.2  # Obese - moderate risk
        
        # Combined risk score calculation
        # Weight: 40% diabetes, 40% cardio, 15% age, 5% BMI
        total_risk_score = (
            diabetes_risk * 0.40 + 
            cardio_risk * 0.40 + 
            age_factor * 0.15 + 
            bmi_factor * 0.05
        )
        
        # Ensure score is between 0 and 1
        total_risk_score = float(np.clip(total_risk_score, 0.0, 1.0))
        
        # Risk categorization based on your specified probability ranges
        if total_risk_score >= 0.7:
            risk_category = "High Risk"
            risk_period = "Immediate medical consultation recommended"
        elif total_risk_score >= 0.3:
            risk_category = "Medium Risk"
            risk_period = "Monitor closely, check-up within 3-6 months"
        else:
            risk_category = "Low Risk"
            risk_period = "Annual check-up recommended"
        
        return {
            'total_risk_score': total_risk_score,
            'risk_category': risk_category,
            'risk_period': risk_period,
            'diabetes_risk': diabetes_risk,
            'cardio_risk': cardio_risk,
            'age_factor': age_factor,
            'bmi_factor': bmi_factor
        }