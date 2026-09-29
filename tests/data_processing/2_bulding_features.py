from src.data_loader.cleaned import load_clean_data
from src.features.build_features import build

df = load_clean_data()
build(df)