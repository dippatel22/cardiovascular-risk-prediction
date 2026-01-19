#!/usr/bin/env python3
"""
Script to create pre-trained models for all available model types.
This will generate .pkl files for each model type for both cardiovascular and diabetes prediction.
"""

import os
import pandas as pd
from ml_models import CardiovascularPredictor, DiabetesPredictor, build_model_candidates

def create_all_pretrained_models():
    """Create pre-trained models for all available model types"""
    print("="*60)
    print(" Creating Pre-trained Models for All Types")
    print("="*60)
    
    # Check if datasets exist
    if not os.path.exists('Cardiovascular_Disease_Dataset.csv'):
        print(" Error: Cardiovascular_Disease_Dataset.csv not found!")
        return
    
    if not os.path.exists('diabetes_train(T2).csv'):
        print(" Error: diabetes_train(T2).csv not found!")
        return
    
    # Create pkls directory if it doesn't exist
    os.makedirs('pkls', exist_ok=True)
    
    # Load datasets
    print("\n Loading datasets...")
    cardio_df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
    diabetes_df = pd.read_csv('diabetes_train(T2).csv')
    
    # Get all available model types
    model_types = list(build_model_candidates().keys())
    
    print(f"\n Training {len(model_types)} model types for both cardiovascular and diabetes prediction...")
    
    for model_type in model_types:
        print(f"\n Training {model_type}...")
        
        try:
            # Train cardiovascular model
            print(f" Training Cardiovascular {model_type}...")
            cardio_trainer = CardiovascularPredictor()
            cardio_results = cardio_trainer.train(cardio_df, model_choice=model_type)
            
            # Save cardiovascular model
            cardio_filename = f"pkls/cardio_{model_type.lower().replace(' ', '_')}.pkl"
            cardio_trainer.save_model(cardio_filename)
            print(f" Saved: {cardio_filename} (Accuracy: {cardio_results['test_accuracy']:.3f})")
            
            # Train diabetes model
            print(f" Training Diabetes {model_type}...")
            diabetes_trainer = DiabetesPredictor()
            diabetes_results = diabetes_trainer.train(diabetes_df, model_choice=model_type)
            
            # Save diabetes model
            diabetes_filename = f"pkls/diabetes_{model_type.lower().replace(' ', '_')}.pkl"
            diabetes_trainer.save_model(diabetes_filename)
            print(f"  Saved: {diabetes_filename} (Accuracy: {diabetes_results['test_accuracy']:.3f})")
            
        except Exception as e:
            print(f"  Error training {model_type}: {str(e)}")
    
    print("\n" + "="*60)
    print(" All pre-trained models created successfully!")
    print("Models are stored in the 'pkls' folder.")
    print("The application will now use these pre-trained models instead of training on startup.")
    print("="*60)

if __name__ == "__main__":
    create_all_pretrained_models()