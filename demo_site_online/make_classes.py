import os, json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = r"C:\Users\Alexandra\Downloads\archive (3)\Respiratory_Sound_Database"
DIAGNOSIS_PATH = os.path.join(BASE_DIR, "patient_diagnosis.csv")

diag_df = pd.read_csv(DIAGNOSIS_PATH)
DIAG_COL = diag_df.columns[1]

class_counts = diag_df[DIAG_COL].value_counts()
classes_ok = class_counts[class_counts >= 3].index.tolist()

filtered = diag_df[diag_df[DIAG_COL].isin(classes_ok)][DIAG_COL].values

le = LabelEncoder()
le.fit(filtered)
classes = le.classes_.tolist()

with open("classes.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print("classes.json creat cu", len(classes), "clase:")
print(classes)
