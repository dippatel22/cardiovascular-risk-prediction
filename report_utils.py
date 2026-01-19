from datetime import datetime
from health_suggestions import generate_personalized_suggestions

def build_report_html(user_info, selected_model, predictions):
    """Build HTML report with all prediction results and suggestions"""
    
    # Get current date
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract user information
    name = user_info.get('name', 'N/A')
    age = user_info.get('age', 'N/A')
    gender = user_info.get('gender', 'N/A')
    email = user_info.get('email', 'N/A')  # Show actual email, not protected
    address = user_info.get('address', 'N/A')
    
    # Extract prediction results
    diabetes_pred = predictions.get('diabetes_prediction', 0)
    diabetes_prob = predictions.get('diabetes_probability', 0)
    cardio_pred = predictions.get('cardiovascular_prediction', 0)
    cardio_prob = predictions.get('cardiovascular_probability', 0)
    
    # Risk assessment
    risk_assessment = predictions.get('risk_assessment', {})
    risk_category = risk_assessment.get('risk_category', 'Unknown')
    risk_period = risk_assessment.get('risk_period', 'Unknown')
    
    # Generate personalized suggestions based on patient data
    patient_data = predictions.get('patient_data', {})
    suggestions = generate_personalized_suggestions(patient_data)
    
    # Format prediction results
    diabetes_result = f"YES - Diabetes (Probability: {diabetes_prob:.3f})" if diabetes_pred == 1 else "NO - No Diabetes"
    cardio_result = f"YES - Cardiovascular Disease (Probability: {cardio_prob:.3f})" if cardio_pred == 1 else "NO - No Cardiovascular Disease"
    
    # Build suggestions HTML
    suggestions_html = ""
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            suggestions_html += f"""
            <div style="margin: 10px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 5px;">
                <h4 style="margin: 0 0 8px 0; color: #1976d2;">{i}. {suggestion['feature']}</h4>
                <p style="margin: 0; font-size: 14px; line-height: 1.4; color: #333;">{suggestion['suggestion']}</p>
            </div>
            """
    else:
        suggestions_html = """
        <div style="margin: 10px 0; padding: 15px; background-color: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="margin: 0 0 8px 0; color: #388e3c;">Excellent Health Profile!</h4>
            <p style="margin: 0; font-size: 14px; line-height: 1.4; color: #333;">Your health parameters are within normal ranges. Continue maintaining a balanced diet, regular physical activity, adequate sleep, and routine check-ups.</p>
        </div>
        """
    
    # Determine risk styling
    if risk_category == "High Risk":
        risk_color = "#f44336"
        risk_bg = "#ffebee"
        risk_icon = "🔴"
    elif risk_category == "Medium Risk":
        risk_color = "#ff9800"
        risk_bg = "#fff3e0"
        risk_icon = "🟡"
    else:
        risk_color = "#4caf50"
        risk_bg = "#e8f5e8"
        risk_icon = "🟢"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cardiovascular Risk Assessment Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 20px;
            }}
            .header h1 {{
                color: #1f77b4;
                margin: 0;
                font-size: 2.2em;
            }}
            .header p {{
                color: #666;
                margin: 10px 0 0 0;
                font-size: 1.1em;
            }}
            .section {{
                margin: 25px 0;
            }}
            .section h2 {{
                color: #ff7f0e;
                border-bottom: 2px solid #ff7f0e;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 15px 0;
            }}
            .info-item {{
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }}
            .info-item strong {{
                color: #333;
            }}
            .prediction-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 5px solid #007bff;
            }}
            .risk-box {{
                background-color: {risk_bg};
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 5px solid {risk_color};
            }}
            .risk-box h3 {{
                color: {risk_color};
                margin-top: 0;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
            .model-info {{
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>❤️ Cardiovascular Risk Assessment Report</h1>
                <p>Generated on {current_date}</p>
            </div>
            
            <div class="section">
                <h2> Patient Information</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Name:</strong> {name}
                    </div>
                    <div class="info-item">
                        <strong>Age:</strong> {age} years
                    </div>
                    <div class="info-item">
                        <strong>Gender:</strong> {gender}
                    </div>
                    <div class="info-item">
                        <strong>Email:</strong> {email}
                    </div>
                </div>
                <div class="info-item" style="margin-top: 15px;">
                    <strong>Address:</strong> {address}
                </div>
            </div>
            
            <div class="section">
                <h2> Model Information</h2>
                <div class="model-info">
                    <strong>Selected Model:</strong> {selected_model or 'Random Forest'}<br>
                    <strong>Analysis Type:</strong> Combined Cardiovascular and Diabetes Risk Assessment
                </div>
            </div>
            
            <div class="section">
                <h2> Prediction Results</h2>
                <div class="prediction-box">
                    <h3>Diabetes Prediction</h3>
                    <p><strong>Result:</strong> {diabetes_result}</p>
                </div>
                <div class="prediction-box">
                    <h3>Cardiovascular Disease Prediction</h3>
                    <p><strong>Result:</strong> {cardio_result}</p>
                </div>
            </div>
            
            <div class="section">
                <h2> Overall Risk Assessment</h2>
                <div class="risk-box">
                    <h3>{risk_icon} Overall Risk Level: {risk_category}</h3>
                    <p><strong>Recommended Monitoring Period:</strong> {risk_period}</p>
                </div>
            </div>
            
            <div class="section">
                <h2> Personalized Health Suggestions</h2>
                {suggestions_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content