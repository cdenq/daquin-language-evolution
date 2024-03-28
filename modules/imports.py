#----------------------------------------------------
# Libraries
#----------------------------------------------------
# EDA / Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams

# Computation / String
import math
import numpy as np
import re

# Formatting / Outputting
from IPython.display import Image
import dataframe_image as dfi

# Data Preprocessing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Data Preprocessing (NLP Vectorization)
nltk.download("vader_lexicon")
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Modeling
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

# Modeling (Evaluation)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, make_scorer

# Deployment
import joblib

#----------------------------------------------------
# Global Variables
#----------------------------------------------------
RANDOM_SEED = 42
CHARS_TO_REMOVE = ["[", "]", "'", '"', "/", "(", ")", "&", ",", ".", ";", "?", "!", "{", "}", "*"]

#----------------------------------------------------
# FILEPATH Variables
#----------------------------------------------------
DEV_PATH_TO_EDA = "../outputs/eda_outputs"
DEV_PATH_TO_MODEL = "../outputs/model_outputs"
DEV_PATH_TO_SAVED_MODELS = "../outputs/saved_models"
DEV_PATH_TO_RAW_DATA = "../data/raw"
DEV_PATH_TO_PREPPED_DATA = "../data/prepped"

APP_PATH_TO_SAVED_MODELS = "outputs/saved_models"
APP_PATH_TO_RAW_DATA = "data/raw"
APP_PATH_TO_PREPPED_DATA = "data/prepped"

#----------------------------------------------------
# Default Variables
#----------------------------------------------------
DEFAULT_GRID_ALPHA = 0.5
DEFAULT_GRAPH_ALPHA = 0.7
DEFAULT_MARKER_SIZE = 8
DEFAULT_BAR_WIDTH = 0.25
DEFAULT_LONG_FIG_SIZE = (6, 4)
DEFAULT_TALL_FIG_SIZE = (4, 6)
DEFAULT_SQUARE_FIG_SIZE = (6, 6)
DEFAULT_BIG_FIG_SIZE = (8, 6)

DEFAULT_TRAIN = 0.75
DEFAULT_VAL = 0.10
DEFAULT_TEST = 0.15
DEFAULT_FOLDS = 5

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()