import os
import csv
import json
import pandas as pd
import itertools
from tqdm import tqdm
from collections import Counter


def create_dict(dataset_path):
	print("creating collection")
	progressive_id = 0
	progressive_id2row_df = {}
	x=[]
	for source in tqdm(os.listdir(dataset_path)):
		for specification in os.listdir(os.path.join(dataset_path, source)):
			with open(os.path.join(dataset_path, source, specification)) as specification_file:
				specification_data = json.load(specification_file)
				x.append(specification_data)
	c = Counter(frozenset(i) for i in x)
	return c

if __name__ == '__main__':
	dataset_path="C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/raw_files/2013_camera_specs"
	my_dict=dict(create_dict(dataset_path))
	with open('C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/dict_schema.csv','w') as f:
		w = csv.writer(f)
		w.writerows(my_dict.items())
