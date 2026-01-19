import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from ml_models import (
    CardiovascularPredictor,
    DiabetesPredictor,
    CombinedRiskPredictor,
    get_model_descriptions,
    build_model_candidates,
)
from explainer import ModelExplainer, HealthSuggestionGenerator
from report_utils import build_report_html
from health_suggestions import generate_personalized_suggestions

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Risk Predictor for Diabetes Patients",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .suggestion-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .feature-impact {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_datasets():
    cardio_df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
    cardio_df_show = cardio_df.drop('patientid', axis=1) if 'patientid' in cardio_df.columns else cardio_df
    diabetes_df = pd.read_csv('diabetes_train(T2).csv')
    return cardio_df, cardio_df_show, diabetes_df

def display_risk_assessment(risk_assessment):
    """Display risk assessment results without risk score graph"""
    st.markdown('<h2 class="sub-header">📊 Risk Assessment Results</h2>', unsafe_allow_html=True)
    
    risk_score = risk_assessment['total_risk_score']
    risk_category = risk_assessment['risk_category']
    risk_period = risk_assessment['risk_period']
    
    # Determine risk level styling
    if risk_category == "High Risk":
        risk_class = "risk-high"
        risk_color = "#f44336"
        risk_icon = "🔴"
    elif risk_category == "Medium Risk":
        risk_class = "risk-medium"
        risk_color = "#ff9800"
        risk_icon = "🟡"
    else:
        risk_class = "risk-low"
        risk_color = "#4caf50"
        risk_icon = "🟢"
    
    # Display risk information with better styling (removed risk score)
    st.markdown(f"""
    <div class="{risk_class}" style="margin: 1rem 0; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0; font-size: 1.5rem; color: black;">{risk_icon} Overall Risk: {risk_category}</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0; color: black;"><strong>Monitoring Period:</strong> {risk_period}</p>
    </div>
    """, unsafe_allow_html=True)

def display_predictions(predictions):
    """Display prediction results - only show probability when disease is predicted (YES)"""
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        diabetes_pred = predictions['diabetes_prediction']
        diabetes_prob = predictions['diabetes_probability']
        
        st.markdown("### Diabetes Prediction")
        if diabetes_pred == 1:  # Disease predicted
            st.error(f"**Predicted: YES - Diabetes** (Probability: {diabetes_prob:.3f})")
        else:  # No disease predicted - don't show probability
            st.success("**Predicted: NO - No Diabetes**")
    
    with col2:
        cardio_pred = predictions['cardiovascular_prediction']
        cardio_prob = predictions['cardiovascular_probability']
        
        st.markdown("### Cardiovascular Disease Prediction")
        if cardio_pred == 1:  # Disease predicted
            st.error(f"**Predicted: YES - Cardiovascular Disease** (Probability: {cardio_prob:.3f})")
        else:  # No disease predicted - don't show probability
            st.success("**Predicted: NO - No Cardiovascular Disease**")

def display_most_important_features(explanation, model_type, has_disease):
    """Display most important features as text instead of diagrams"""
    st.markdown(f'<h3 class="sub-header">{model_type} - Key Health Factors</h3>', unsafe_allow_html=True)
    
    if 'feature_importance' in explanation and has_disease:
        # Get top 3 most important features
        features = explanation['feature_importance'][:3]
        
        st.markdown(f"**The following factors most affect your {model_type.lower()} health:**")
        features = [f for f in explanation['feature_importance'] if f['importance'] > 0]
        for i, feature_info in enumerate(features, 1):
            feature_name = feature_info['feature']
            importance = abs(feature_info['importance'])
            impact_direction = "increases"
            
            # Create a more readable feature name
            readable_name = feature_name.replace('_', ' ').title()
            
            st.markdown(f"""
            <div class="feature-impact">
                <h4 style="margin: 0 0 0.5rem 0; color: #007bff;">{i}. {readable_name}</h4>
                <p style="margin: 0; color: black;">This factor significantly {impact_direction} your risk </p>
            </div>
            """, unsafe_allow_html=True)
    elif not has_disease:
        st.markdown(f"""
        <div class="feature-impact">
            <p style="margin: 0; color: #28a745; font-weight: bold;"> No significant risk factors detected for {model_type.lower()}. Maintain your current healthy lifestyle!</p>
        </div>
        """, unsafe_allow_html=True)

def display_suggestions(patient_data):
    """Display personalized health suggestions based on input values"""
    st.markdown('<h2 class="sub-header"> Personalized Health Suggestions</h2>', unsafe_allow_html=True)
    
    suggestions = generate_personalized_suggestions(patient_data)
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"""
            <div class="suggestion-box" style="margin: 0.5rem 0; padding: 1rem; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 5px;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">{i}. {suggestion['feature']}</h4>
                <p style="margin: 0; font-size: 1rem; color: black; line-height: 1.4;">{suggestion['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="suggestion-box" style="margin: 0.5rem 0; padding: 1rem; background-color: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="margin: 0 0 0.5rem 0; color: #388e3c;">Excellent Health Profile!</h4>
            <p style="margin: 0; font-size: 1rem; color: black; line-height: 1.4;">Your health parameters are within normal ranges. Continue maintaining a balanced diet, regular physical activity, adequate sleep, and routine check-ups.</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data
def load_model_performance_data():
    """Load model performance data from pre-trained models"""
    cardio_df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
    diabetes_df = pd.read_csv('diabetes_train(T2).csv')
    
    # Train all models once to get performance metrics (but don't save them)
    import importlib, ml_models as _mlm
    _mlm = importlib.reload(_mlm)
    
    # Train cardiovascular models
    cardio_trainer = _mlm.CardiovascularPredictor()
    train_cv_info = cardio_trainer.train(cardio_df)
    
    # Train diabetes models
    diabetes_trainer = _mlm.DiabetesPredictor()
    train_db_info = diabetes_trainer.train(diabetes_df)
    
    return {
        'ranked_cv': train_cv_info['ranked_models'],
        'ranked_db': train_db_info['ranked_models']
    }

def load_pretrained_model(model_choice):
    """Load pre-trained models from pkls folder"""
    # Generate filenames based on model choice
    cardio_filename = f"pkls/cardio_{model_choice.lower().replace(' ', '_')}.pkl"
    diabetes_filename = f"pkls/diabetes_{model_choice.lower().replace(' ', '_')}.pkl"
    
    # Check if files exist
    if not os.path.exists(cardio_filename):
        raise FileNotFoundError(f"Pre-trained cardiovascular model not found: {cardio_filename}")
    if not os.path.exists(diabetes_filename):
        raise FileNotFoundError(f"Pre-trained diabetes model not found: {diabetes_filename}")
    
    # Load models
    cardio_predictor = CardiovascularPredictor()
    cardio_predictor.load_model(cardio_filename)
    
    diabetes_predictor = DiabetesPredictor()
    diabetes_predictor.load_model(diabetes_filename)
    
    return cardio_predictor, diabetes_predictor

def main():
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Risk Predictor for Diabetes Patients</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Load model performance data once at startup (for display purposes only)
    if 'model_results' not in st.session_state:
        with st.spinner("Loading model performance data... Please wait."):
            st.session_state.model_results = load_model_performance_data()
            st.session_state.ranked_cv = st.session_state.model_results['ranked_cv']
            st.session_state.ranked_db = st.session_state.model_results['ranked_db']
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation Steps")
        steps = [
            "1️⃣ User Info",
            "2️⃣ Model Selection",
            "3️⃣ Cardio Inputs",
            "4️⃣ Diabetes Inputs",
            "5️⃣ Results & Report",
        ]
        for idx, label in enumerate(steps, start=1):
            if st.session_state.step == idx:
                st.markdown(f"**➡️ {label}**")
            else:
                st.markdown(f"{label}")
        st.markdown("---")
        
        # Quick navigation buttons
        st.markdown("### Quick Navigation")
        if st.button(" Start Over"):
            for key in list(st.session_state.keys()):
                if key not in ['model_results', 'ranked_cv', 'ranked_db']:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()

    # Pages
    if st.session_state.step == 1:
        # User info
        st.markdown('<h2 class="sub-header">👤 Basic User Information</h2>', unsafe_allow_html=True)
        
        # Use form to prevent auto-submission
        with st.form("user_info_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name *", value=st.session_state.get('user_name', ''))
                age = st.number_input("Age *", min_value=1, max_value=120, value=st.session_state.get('user_age', 40))
                gender = st.selectbox("Gender *", ["Male", "Female"], index=0 if st.session_state.get('user_gender', 'Male') == 'Male' else 1)
            with col2:
                email = st.text_input("Email *", value=st.session_state.get('user_email', ''))
                address = st.text_area("Address *", height=100, value=st.session_state.get('user_address', ''))
            
            submitted = st.form_submit_button("Next Step ➡️", use_container_width=True)
            
        if submitted:
            # Validation
            errors = []
            if not name or name.strip() == "":
                errors.append("Full Name is required")
            if not email or email.strip() == "":
                errors.append("Email is required")
            elif "@" not in email or "." not in email.split("@")[-1]:
                errors.append("Please enter a valid email format")
            if not address or address.strip() == "":
                errors.append("Address is required")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Store in session state
                st.session_state.user_name = name
                st.session_state.user_age = age
                st.session_state.user_gender = gender
                st.session_state.user_email = email
                st.session_state.user_address = address
                st.session_state.user_info = {"name": name, "age": age, "gender": gender, "email": email, "address": address}
                st.session_state.step = 2
                st.success(" User information saved!")
                st.rerun()
                
    elif st.session_state.step == 2:
        # Model selection
        st.markdown('<h2 class="sub-header"> Model Selection</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("###  Model Performance")
            st.markdown("**Cardiovascular Models:**")
            cv_df = pd.DataFrame(st.session_state.get('ranked_cv', []))
            if not cv_df.empty:
                cv_df = cv_df.round(3)
                st.dataframe(cv_df, use_container_width=True)
            
            st.markdown("**Diabetes Models:**")
            db_df = pd.DataFrame(st.session_state.get('ranked_db', []))
            if not db_df.empty:
                db_df = db_df.round(3)
                st.dataframe(db_df, use_container_width=True)
            
            model_names = list(build_model_candidates().keys())
            choice = st.radio("Choose model:", model_names, index=0)
            
        with col2:
            st.markdown("###  Model Descriptions")
            desc = get_model_descriptions()
            for name in build_model_candidates().keys():
                with st.expander(f"ℹ️ {name}"):
                    st.write(desc.get(name, ''))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        
        with col3:
            if st.button("Select & Continue ➡️", use_container_width=True):
                with st.spinner('Loading selected pre-trained model...'):
                    try:
                        # Load pre-trained models from pkls folder
                        cardio_predictor, diabetes_predictor = load_pretrained_model(choice)
                        
                        st.session_state.cardio_predictor = cardio_predictor
                        st.session_state.diabetes_predictor = diabetes_predictor
                        st.session_state.selected_model = choice
                        st.session_state.step = 3
                        st.success(f" {choice} pre-trained model loaded successfully!")
                        st.rerun()
                        
                    except FileNotFoundError as e:
                        st.error(f" Pre-trained model not found: {str(e)}")
                        st.error("Please run 'python create_pretrained_models.py' first to create the models.")
                    except Exception as e:
                        st.error(f" Error loading model: {str(e)}")
                    
    elif st.session_state.step == 3:
        # Cardio inputs - COMPLETELY ISOLATED IN FORM
        st.markdown('<h2 class="sub-header">❤️ Cardiovascular Information</h2>', unsafe_allow_html=True)
        
        # CRITICAL: Everything inside this form - NO external state updates
        with st.form("cardio_form", clear_on_submit=False):
            st.markdown("**Fill in your cardiovascular health information below. Click 'Next Step' when complete.**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Blood Pressure & Heart")
                resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=300, 
                                           value=st.session_state.get('cardio_resting_bp', 120))
                max_heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, 
                                               value=st.session_state.get('cardio_max_heart_rate', 150))
                
                st.markdown("#### Cholesterol & Blood Sugar")
                serum_cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, 
                                                  value=st.session_state.get('cardio_serum_cholesterol', 200))
                fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"],
                                                 index=0 if st.session_state.get('cardio_fasting_blood_sugar', 'No') == 'No' else 1)
                
            with col2:
                st.markdown("#### Symptoms & Tests")
                chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
                chest_pain = st.selectbox("Chest Pain Type", chest_pain_options,
                                        index=st.session_state.get('cardio_chest_pain_idx', 0))
                exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"],
                                             index=0 if st.session_state.get('cardio_exercise_angina', 'No') == 'No' else 1)
                
                st.markdown("#### Medical Results")
                resting_electro_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
                resting_electro = st.selectbox("Resting ECG Results", resting_electro_options,
                                             index=st.session_state.get('cardio_resting_electro_idx', 0))
                oldpeak = st.number_input("ST Depression (Exercise)", min_value=0.0, max_value=10.0, 
                                        value=st.session_state.get('cardio_oldpeak', 0.0), step=0.1)
                slope_options = ["Upsloping", "Flat", "Downsloping"]
                slope = st.selectbox("ST Segment Slope", slope_options,
                                   index=st.session_state.get('cardio_slope_idx', 0))
                major_vessels = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4, 
                                              value=st.session_state.get('cardio_major_vessels', 0))
            
            # Form submission buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                back_clicked = st.form_submit_button("⬅️ Back", use_container_width=True)
            with col3:
                next_clicked = st.form_submit_button("Next Step ➡️", use_container_width=True, type="primary")
        
        # Handle form submissions ONLY
        if back_clicked:
            st.session_state.step = 2
            st.rerun()
            
        if next_clicked:
            # Save all cardio data to session state
            st.session_state.cardio_resting_bp = resting_bp
            st.session_state.cardio_max_heart_rate = max_heart_rate
            st.session_state.cardio_serum_cholesterol = serum_cholesterol
            st.session_state.cardio_fasting_blood_sugar = fasting_blood_sugar
            st.session_state.cardio_chest_pain = chest_pain
            st.session_state.cardio_chest_pain_idx = chest_pain_options.index(chest_pain)
            st.session_state.cardio_exercise_angina = exercise_angina
            st.session_state.cardio_resting_electro = resting_electro
            st.session_state.cardio_resting_electro_idx = resting_electro_options.index(resting_electro)
            st.session_state.cardio_oldpeak = oldpeak
            st.session_state.cardio_slope = slope
            st.session_state.cardio_slope_idx = slope_options.index(slope)
            st.session_state.cardio_major_vessels = major_vessels
            
            # Prepare data for model
            gender_encoded = 1 if st.session_state.get('user_info', {}).get("gender", "Male") == "Male" else 0
            age_val = int(st.session_state.get('user_info', {}).get("age", 40))
            
            mapping_chest = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            mapping_rest = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            
            st.session_state.patient_cardio = {
                'age': age_val,
                'gender': gender_encoded,
                'chestpain': mapping_chest[chest_pain],
                'restingBP': resting_bp,
                'serumcholestrol': serum_cholesterol,
                'fastingbloodsugar': 1 if fasting_blood_sugar == "Yes" else 0,
                'restingrelectro': mapping_rest[resting_electro],
                'maxheartrate': max_heart_rate,
                'exerciseangia': 1 if exercise_angina == "Yes" else 0,
                'oldpeak': oldpeak,
                'slope': {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope],
                'noofmajorvessels': major_vessels,
            }
            st.session_state.step = 4
            st.success(" Cardiovascular data saved!")
            st.rerun()
            
    elif st.session_state.step == 4:
        # Diabetes inputs - COMPLETELY ISOLATED IN FORM
        st.markdown('<h2 class="sub-header">🩺 Diabetes Information</h2>', unsafe_allow_html=True)
        
        with st.form("diabetes_form", clear_on_submit=False):
            st.markdown("**Fill in your diabetes-related health information below. Click 'Predict Risk' when complete.**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Basic Health Metrics")
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, 
                                            value=st.session_state.get('diabetes_pregnancies', 0))
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=500, 
                                        value=st.session_state.get('diabetes_glucose', 100))
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=300, 
                                               value=st.session_state.get('diabetes_blood_pressure', 80))
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, 
                                               value=st.session_state.get('diabetes_skin_thickness', 20))
                
            with col2:
                st.markdown("#### Advanced Metrics")
                insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=1000, 
                                        value=st.session_state.get('diabetes_insulin', 100))
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=100.0, 
                                    value=st.session_state.get('diabetes_bmi', 25.0), step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, 
                                    value=st.session_state.get('diabetes_dpf', 0.5), step=0.01)
                
                age_val = int(st.session_state.get('user_info', {}).get("age", 40))
                st.info(f"Age from user info: {age_val} years")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                back_clicked = st.form_submit_button("⬅️ Back", use_container_width=True)
            with col3:
                predict_clicked = st.form_submit_button(" Predict Risk ➡️", use_container_width=True, type="primary")
        
        if back_clicked:
            st.session_state.step = 3
            st.rerun()
            
        if predict_clicked:
            # Save diabetes data to session state
            st.session_state.diabetes_pregnancies = pregnancies
            st.session_state.diabetes_glucose = glucose
            st.session_state.diabetes_blood_pressure = blood_pressure
            st.session_state.diabetes_skin_thickness = skin_thickness
            st.session_state.diabetes_insulin = insulin
            st.session_state.diabetes_bmi = bmi
            st.session_state.diabetes_dpf = dpf
            
            # Save diabetes data and make predictions
            st.session_state.patient_diabetes = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age_val,
            }
            
            with st.spinner('Making predictions using pre-trained models...'):
                patient_data = {**st.session_state.patient_diabetes, **st.session_state.patient_cardio}
                combined = CombinedRiskPredictor()
                combined.cardio_predictor = st.session_state.get('cardio_predictor')
                combined.diabetes_predictor = st.session_state.get('diabetes_predictor')
                preds = combined.predict_for_diabetes_patient(patient_data)
                
                # Generate explanations
                explainer = ModelExplainer(st.session_state.get('cardio_predictor'), st.session_state.get('diabetes_predictor'))
                try:
                    cardio_df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
                    if 'patientid' in cardio_df.columns:
                        cardio_df = cardio_df.drop('patientid', axis=1)
                    cardio_X = cardio_df.drop('target', axis=1)
                    diabetes_df = pd.read_csv('diabetes_train(T2).csv')
                    diabetes_X = diabetes_df.drop('Outcome', axis=1)
                    explainer.setup_lime_explainers(cardio_X, diabetes_X)
                except Exception:
                    pass
                
                diabetes_expl = explainer.explain_diabetes_prediction(st.session_state.patient_diabetes, method='lime')
                cardio_expl = explainer.explain_cardio_prediction(st.session_state.patient_cardio, method='lime')
                
                preds['diabetes_explanation'] = diabetes_expl
                preds['cardio_explanation'] = cardio_expl
                preds['patient_data'] = patient_data  # Store for suggestions
                st.session_state.predictions = preds
                st.session_state.step = 5
                st.success(" Predictions completed!")
                st.rerun()
                
    else:
        # Results page
        preds = st.session_state.get('predictions')
        if preds is None:
            st.error("No predictions available. Please start over.")
            if st.button("Start Over"):
                st.session_state.step = 1
                st.rerun()
            return
            
        display_predictions(preds)
        display_risk_assessment(preds['risk_assessment'])
        
        st.markdown('<h2 class="sub-header"> Key Health Factors Analysis</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            has_diabetes = preds['diabetes_prediction'] == 1
            display_most_important_features(preds['diabetes_explanation'], "Diabetes", has_diabetes)
        with col2:
            has_cardio = preds['cardiovascular_prediction'] == 1
            display_most_important_features(preds['cardio_explanation'], "Cardiovascular", has_cardio)
            
        # Display personalized suggestions based on input values
        display_suggestions(preds.get('patient_data', {}))
        
        # Report download
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(" New Prediction", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['model_results', 'ranked_cv', 'ranked_db']:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()
        
        with col3:
            html = build_report_html(st.session_state.get('user_info', {}), st.session_state.get('selected_model'), preds)
            st.download_button(
                " Download Report", 
                data=html.encode('utf-8'), 
                file_name="cardiovascular_risk_report.html", 
                mime="text/html",
                use_container_width=True
            )

if __name__ == "__main__":
    main()