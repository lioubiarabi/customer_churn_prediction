from src.data_loader.raw import load_raw_data
from src.cleaner.clean import clean

df = load_raw_data()
print(clean(df))