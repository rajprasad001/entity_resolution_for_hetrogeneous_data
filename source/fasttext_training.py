import fasttext
import pandas as pd
from datetime import datetime

model = fasttext.train_unsupervised('C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/text_files/2020-04-06  22-39-cleaned_dataframe_text.txt', minn=2, maxn=15, dim=300, model='skipgram',epoch =10000, verbose=2)
filename1 = datetime.now().strftime("%Y-%m-%d  %H-%M")
timestamp_string = str(filename1)
model.save_model(f"C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/bin_files/{timestamp_string}-skipgram_embedding.bin")
