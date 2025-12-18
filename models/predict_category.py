from pathlib import Path
import joblib 
BASE_DIR=Path(__file__).resolve().parent
MODEL_PATH=BASE_DIR /"final_model.pkl"
model=joblib.load(MODEL_PATH)
def main():
    print("Model se ucitava...")
    model=joblib.load(MODEL_PATH)
    print("Model uspjesno ucitan")
    while True:
        text=input("\nUnesi naziv proizvoda(ili ˙exit˙ za izlaz): ")
        if text.lower() == "exit":
            print("Izlaz iz programa.")
            break
        prediction= model.predict([text])
        print(f"Predvidjena kategorija: {prediction[0]}")
if __name__ == "__main__":
  main()