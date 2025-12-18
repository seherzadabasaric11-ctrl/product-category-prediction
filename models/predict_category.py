import joblib 
MODEL_PATH="product_category_model.pkl"
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