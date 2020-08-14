import pandas as pd
import numpy as np

raw_df = pd.read_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/cleaned_dataframe-200405.csv")

#Generating binary code for models _____________________________________________________________________________________________________________________________________________________________________
list_of_models = sorted(list(set([str(x).strip() for x in raw_df['model'].to_list()])))

list_size = len(list_of_models)
print(list_size)


model_to_code=dict()
num=1
for model in list_of_models:
  binary_num='{0:0b}'.format(num)
  while (len(binary_num)<14):
    binary_num = "0" + binary_num
  num+=1
  model_to_code[model]=binary_num
print(model_to_code)
counter=0
for key in model_to_code.keys():
  #print(key)
  counter+=1
  if counter>2000:
    break

for index,model in enumerate(raw_df['model']):
    for key in model_to_code.keys():
        if model ==key:
            raw_df.at[index,'model_binary']=model_to_code.get(key)

#Generating binary code for models ____________________________________________________________________________________________________________________________________________________________________
list_of_brands = sorted(list(set([str(x).strip() for x in raw_df['brand'].to_list()])))

list_size2 = len(list_of_brands)
print(list_size2)

brand_to_code=dict()
num2=1
for brand in list_of_brands:
  binary_num2='{0:0b}'.format(num2)
  while (len(binary_num2)<10):
    binary_num2 = "0" + binary_num2
  num2+=1
  brand_to_code[brand]=binary_num2
print(brand_to_code)
counter2=0
for key in model_to_code.keys():
  #print(key)
  counter2+=1
  if counter2>2000:
    break

for index,brand in enumerate(raw_df['brand']):
    for key in brand_to_code.keys():
        if brand ==key:
            raw_df.at[index,'brand_binary']=brand_to_code.get(key)


raw_df.to_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/binary_codes.csv",index = False)


