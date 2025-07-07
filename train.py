# data
import pandas as pd
import numpy as np

# visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# pemprosesan & pemodelan data
import time
import sklearn
import skops.io as sio
from skops.io import dump, load, get_untrusted_types
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import minmax_scale, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

data = pd.read_csv(f"Data/credit_risk_dataset.csv")

data.dropna(axis=0,inplace=True)
data = data.drop_duplicates()
data.reset_index(inplace = True)
data = data.drop(data[data['person_age'] > 55].index, axis=0)
data = data.drop(data[data['person_emp_length'] > 38].index, axis=0)
data = data.drop(['index'], axis=1)
data.reset_index(inplace = True)
data = data.drop(['index'], axis=1)

# Mengonversi Data Kategorikal menjadi Numerik (Label Encoding)
cat_columns = ['cb_person_default_on_file', 'person_home_ownership','loan_intent']

default_on_file = LabelEncoder()
default_on_file.fit( data[cat_columns[0]])

home = LabelEncoder()
home.fit( data[cat_columns[1]] )

loan = LabelEncoder()
loan.fit( data[cat_columns[2]] )

# Konversi data kategorikal menjadi data numerik
data[cat_columns[0] ] = data[cat_columns[0] ].apply( lambda x: default_on_file.transform([x])[0] )
data[cat_columns[1] ] = data[cat_columns[1] ].apply( lambda x: home.transform([x])[0] )
data[cat_columns[2] ] = data[cat_columns[2] ].apply( lambda x: loan.transform([x])[0] )

cleaned_data = data.copy()

rus = RandomUnderSampler(random_state=42)

X, y = rus.fit_resample( cleaned_data.drop(["loan_status"], axis=1), cleaned_data["loan_status"])
X = X.drop(['loan_grade'], axis=1)

# Pembagian Data untuk Pelatihan dan Pengujian
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42)

from sklearn.metrics import accuracy_score, f1_score

model = Pipeline(
    steps=[
        ("model", ExtraTreesClassifier())
    ]
)
model.fit(x_train, y_train)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

# uji model
predict = model.predict(x_test)
akurasi = accuracy_score(y_test, predict)
f1 = f1_score(y_test, predict, average='macro')

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(akurasi, 2)}, F1 Score = {round(f1, 2)}.")

cm = confusion_matrix(y_test, predict, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = model.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

sio.dump(model, "Model/credit_loan_detection.skops")

unknown_types = get_untrusted_types(file="Model/credit_loan_detection.skops")