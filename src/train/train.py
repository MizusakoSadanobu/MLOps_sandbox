import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRACKING_URI = "https://dagshub.com/<DagsHub-user-name>/<repository-name>.mlflow"
EXPERIMENT_NAME = "LogisticRegression"

