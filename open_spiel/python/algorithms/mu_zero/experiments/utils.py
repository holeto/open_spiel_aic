import pickle

def load_model(filepath: str):
  with open(filepath, "rb") as f:
    data= pickle.load(f)
  return data

def save_model(filepath: str, data):
   with open(filepath, "wb") as f:
    pickle.dump(data, f)
