import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#Ucitavanje podataka
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
data_path=BASE_DIR/"data"/"products.csv"
data=pd.read_csv(data_path)
data=data.dropna(subset=["Product Title"," Category Label"])
X=data ["Product Title"]
y=data[" Category Label"]


#Podjela na train/test
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#TF-IDF + Linear SVM
model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,stop_words="english",max_features=5000)),("svm",LinearSVC())])

#Treniranje
model.fit(X_train,y_train)

#Evaluacija
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model accuracy:{accuracy:.4f}")

#Sacuvaj model
BASE_DIR=Path(__file__).resolve().parent.parent
MODELS_DIR=BASE_DIR/"models"
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR /"product_category_model.pkl")
print("Model uspjesno sacuvan u models/product_category_model.pkl")

#FINAL MODEL (TF-IDF + Logistic Regression)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

final_model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,stop_words="english",max_features=5000)),("clf",LogisticRegression(max_iter=1000))])

#Treniranje finalnog modela 
final_model.fit(X_train,y_train)

#Evaluacija
y_pred_final=final_model.predict(X_test)
print("\nFINAL MODEL RESULTS")
print("Accuracy:",accuracy_score(y_test,y_pred_final))
print("\nClassification report:\n")
print(classification_report(y_test,y_pred_final,zero_division=0))

#Spremanje FINALNOG modela 
joblib.dump(final_model,MODELS_DIR/"final_model.pkl")
print("\n FINAL model sacuvan u models/final_model.pkl")
