import os
import json
import itertools
import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime

def creating_dataframe(path_to_json):
    column_df = ['spec_id', 'page_title']
    progressive_id = 0
    progressive_id2row_df = {}
    for websites in tqdm(os.listdir(path_to_json)):
        #print(websites)
        for files in os.listdir(os.path.join(path_to_json, websites)):

            #print(files)
            file_number = files.replace('.json', '')
            #print(file_number)
            file_id = '{}//{}'.format(websites, file_number)
            #print(file_id)

            with open(os.path.join(path_to_json, websites, files)) as specification_file:
                file_data = json.load(specification_file)
                #print(file_data)
                page_title = file_data.get('<page title>').lower()
                #print(page_title)
                row = (file_id,page_title )
                #print(row)
                progressive_id2row_df.update({progressive_id:row})
                #print(progressive_id2row_df)
                progressive_id += 1
                #print(progressive_id)
    raw_data_dataframe = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=column_df)
    raw_data_dataframe.to_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/raw_data_dataframe.csv", index=None)
 #   data_preprocessing(raw_data_dataframe)
    return (raw_data_dataframe)

def main():

    path_to_json = 'C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/raw_files/2013_camera_specs'
    creating_dataframe(path_to_json)

if __name__ == '__main__':
    main()