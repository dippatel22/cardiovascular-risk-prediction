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