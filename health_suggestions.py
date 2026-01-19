
def generate_personalized_suggestions(patient_data):
    """Generate personalized health suggestions based on actual input values"""
    suggestions = []
    
    # Get patient data values
    bmi = patient_data.get('BMI', 0)
    glucose = patient_data.get('Glucose', 0)
    blood_pressure = patient_data.get('BloodPressure', 0)
    resting_bp = patient_data.get('restingBP', 0)
    cholesterol = patient_data.get('serumcholestrol', 0)
    insulin = patient_data.get('Insulin', 0)
    age = patient_data.get('age', patient_data.get('Age', 0))
    pregnancies = patient_data.get('Pregnancies', 0)
    skin_thickness = patient_data.get('SkinThickness', 0)
    max_heart_rate = patient_data.get('maxheartrate', 0)

    # Extra heart/CVD-related features
    chestpain = patient_data.get('chestpain', 0)
    resting_relectro = patient_data.get('restingrelectro', 0)
    exercise_angina = patient_data.get('exerciseangia', 0)
    oldpeak = patient_data.get('oldpeak', 0)
    slope = patient_data.get('slope', 0)
    no_of_vessels = patient_data.get('noofmajorvessels', 0)
    diabetes_pedigree = patient_data.get('DiabetesPedigreeFunction', 0)
    
    # ---------------------------
    # Metabolic risk factors
    # ---------------------------
    
    # BMI suggestions
    if bmi > 30:
        suggestions.append({
            'feature': 'High BMI (Obesity)',
            'suggestion': 'Your BMI indicates obesity. Focus on regular exercise (150 minutes moderate activity per week), maintain a balanced diet with portion control, and consider consulting a nutritionist for a personalized weight loss plan.'
        })
    elif bmi > 25:
        suggestions.append({
            'feature': 'Elevated BMI (Overweight)',
            'suggestion': 'Your BMI is in the overweight range. Incorporate daily physical activity, reduce caloric intake by 300-500 calories per day, and focus on whole foods while limiting processed foods.'
        })
    elif bmi < 18.5:
        suggestions.append({
            'feature': 'Low BMI (Underweight)',
            'suggestion': 'Your BMI is below normal range. Focus on nutrient-dense, calorie-rich foods, consider strength training to build muscle mass, and consult a healthcare provider to rule out underlying conditions.'
        })
    
    # Glucose suggestions
    if glucose > 140:
        suggestions.append({
            'feature': 'High Blood Glucose',
            'suggestion': 'Your glucose level is elevated. Follow a low-glycemic diet, limit refined sugars and carbohydrates, exercise regularly to improve insulin sensitivity, and monitor blood sugar levels daily.'
        })
    elif glucose > 100:
        suggestions.append({
            'feature': 'Elevated Blood Glucose',
            'suggestion': 'Your glucose level is above normal. Reduce sugar intake, choose complex carbohydrates over simple sugars, maintain regular meal times, and increase physical activity.'
        })
    
    # Blood Pressure suggestions
    bp_to_check = max(blood_pressure, resting_bp)
    if bp_to_check > 140:
        suggestions.append({
            'feature': 'High Blood Pressure',
            'suggestion': 'Your blood pressure is elevated. Reduce sodium intake to less than 2300mg daily, engage in regular aerobic exercise, manage stress through relaxation techniques, and limit alcohol consumption.'
        })
    elif bp_to_check > 120:
        suggestions.append({
            'feature': 'Elevated Blood Pressure',
            'suggestion': 'Your blood pressure is above optimal. Maintain a heart-healthy diet rich in fruits and vegetables, exercise regularly, limit caffeine intake, and practice stress management techniques.'
        })
    
    # Cholesterol suggestions
    if cholesterol > 240:
        suggestions.append({
            'feature': 'High Cholesterol',
            'suggestion': 'Your cholesterol level is high. Adopt a low-saturated fat diet, increase fiber intake through whole grains and vegetables, exercise regularly, and consider omega-3 rich foods like fish.'
        })
    elif cholesterol > 200:
        suggestions.append({
            'feature': 'Elevated Cholesterol',
            'suggestion': 'Your cholesterol is above optimal. Limit saturated fats, choose lean proteins, increase soluble fiber intake, and maintain regular physical activity.'
        })
    
    # Insulin suggestions
    if insulin > 200:
        suggestions.append({
            'feature': 'High Insulin Levels',
            'suggestion': 'Your insulin levels are elevated. Focus on a low-carbohydrate diet, practice intermittent fasting (with medical supervision), engage in resistance training, and maintain consistent meal timing.'
        })
    elif insulin > 100:
        suggestions.append({
            'feature': 'Elevated Insulin',
            'suggestion': 'Your insulin levels are above normal. Reduce refined carbohydrate intake, eat smaller frequent meals, increase physical activity, and consider foods with low glycemic index.'
        })
    
    # Age-related suggestions
    if age > 60:
        suggestions.append({
            'feature': 'Age-Related Health',
            'suggestion': 'As you age, focus on maintaining bone density through weight-bearing exercises, ensure adequate calcium and vitamin D intake, get regular health screenings, and stay socially active.'
        })
    elif age > 45:
        suggestions.append({
            'feature': 'Middle-Age Health',
            'suggestion': 'At your age, prioritize preventive care with regular check-ups, maintain muscle mass through strength training, monitor cardiovascular health, and manage stress effectively.'
        })
    
    # Heart rate suggestions
    if max_heart_rate < 100:
        suggestions.append({
            'feature': 'Low Maximum Heart Rate',
            'suggestion': 'Your maximum heart rate is low. Gradually increase cardiovascular exercise intensity, consider interval training, ensure adequate rest and recovery, and consult a cardiologist if concerned.'
        })
    
    # Pregnancy-related suggestions
    if pregnancies > 4:
        suggestions.append({
            'feature': 'Multiple Pregnancies History',
            'suggestion': 'With multiple pregnancies, monitor for gestational diabetes risk, maintain healthy weight between pregnancies, ensure adequate nutrition, and have regular gynecological check-ups.'
        })

    # ---------------------------
    # Extra CVD / diabetes features
    # ---------------------------

    # Chest Pain
    if chestpain in [1, 2]:  # assuming 1=typical angina, 2=atypical angina
        suggestions.append({
            'feature': 'Chest Pain',
            'suggestion': 'Chest pain during exertion may indicate coronary artery disease. A cardiology evaluation is recommended.'
        })

    # ECG abnormalities
    if resting_relectro != 0 or slope != 0 or oldpeak > 1:
        suggestions.append({
            'feature': 'ECG Abnormalities',
            'suggestion': 'ECG changes suggest possible ischemia. Further cardiac evaluation (stress test, echocardiogram) is advised.'
        })

    # Exercise Angina
    if exercise_angina == 1:
        suggestions.append({
            'feature': 'Exercise-Induced Angina',
            'suggestion': 'Angina during exercise is a strong sign of heart disease risk. Seek medical consultation for further evaluation.'
        })

    # Number of Major Vessels
    if no_of_vessels > 0:
        suggestions.append({
            'feature': 'Blocked Coronary Vessels',
            'suggestion': f'{no_of_vessels} major vessel(s) affected. This significantly increases heart disease risk. Medical management or intervention may be required.'
        })

    # Skin Thickness
    if skin_thickness > 30:
        suggestions.append({
            'feature': 'High Skinfold Thickness',
            'suggestion': 'Excess fat around the abdomen increases insulin resistance and heart disease risk. Focus on weight management and lifestyle changes.'
        })

    # Diabetes Pedigree Function
    if diabetes_pedigree > 0.5:
        suggestions.append({
            'feature': 'Family History of Diabetes',
            'suggestion': 'You have a strong family history of diabetes. Adopt preventive lifestyle measures early and get regular screenings.'
        })
    
    return suggestions
