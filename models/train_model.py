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