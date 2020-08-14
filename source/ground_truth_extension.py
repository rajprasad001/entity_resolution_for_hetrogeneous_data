import csv
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from ast import literal_eval
from datetime import datetime
column_names = ['left_spec_id','right_spec_id']
unique_list = []
old_extended_gt = pd.DataFrame(columns= column_names)
new_extended_gt = pd.DataFrame(columns= column_names)


def main():

    global unique_list,old_extended_gt, new_extended_gt
    # 1. Loading large and medium Ground_truths and then concatinating both the dataset.-----------
    lrg_gt_df = pd.read_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/sigmod_large_labelled_dataset.csv")
    med_gt_df = pd.read_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/sigmod_medium_labelled_dataset.csv")
    comp_gt_df = pd.concat([lrg_gt_df, med_gt_df])

    # 2. Selecting only matched pairs from the concatinated dataframe , i.e., 'comp_gt_dataset'----
    matched_pair_temp_df = comp_gt_df[comp_gt_df['label'] == 1]

    left_list = list(matched_pair_temp_df['left_spec_id'].unique())
    right_list = list(matched_pair_temp_df['right_spec_id'].unique())
    unique_list = list(set(left_list) | set(right_list))

    # 3. Removing 'Label' column.
    merged_data = matched_pair_temp_df.drop(['label'], axis=1)
    i= 0
    assertion_similarity_threshold = 0
    assertion_similarity_threshold_old = 10

    # Checking if at next iteration no, of rows are increasing or not
    while (assertion_similarity_threshold - assertion_similarity_threshold_old != 0):
        if i is 0:
            new_extended_gt = layer(merged_data)
            i+=1
        else:
            new_extended_gt = layer(old_extended_gt)
            i+=1
        old_extended_gt = new_extended_gt
        filename1 = datetime.now().strftime("%Y-%m-%d  %H-%M")
        timestamp_string1 = str(filename1)
        new_extended_gt.to_csv(f"C:/Users/anujp/Desktop/Augmented_Data/{timestamp_string1}key_value_pair_iter{i}.csv")
        #assertion_similarity_threshold = jaccard_similarity_score(new_extended_gt[column_names], old_extended_gt[column_names])
        assertion_similarity_threshold_old = assertion_similarity_threshold
        assertion_similarity_threshold = len(new_extended_gt)

    # Additional check to compare similarity of two dataframes , doesnt compare pairwise similarity
    i=0
    print(jaccard_score(new_extended_gt['left_spec_id'],old_extended_gt['left_spec_id'], average='weighted'))
    print(jaccard_score(new_extended_gt['right_spec_id'],old_extended_gt['right_spec_id'], average='weighted'))
    while i==0:
        old_extended_gt = new_extended_gt
        new_extended_gt = layer(old_extended_gt)
        filename2 = datetime.now().strftime("%Y-%m-%d  %H-%M")
        timestamp_string2 = str(filename2)
        new_extended_gt.to_csv(f"C:/Users/anujp/Desktop/Augmented_Data/{timestamp_string2}key_value_pair_jaccard_iter{i}.csv")
        i += 1
        if (jaccard_score(new_extended_gt['left_spec_id'], old_extended_gt['left_spec_id'],
                             average='weighted') != 1.0 and
               jaccard_score(new_extended_gt['right_spec_id'], old_extended_gt['right_spec_id'],
                             average='weighted') != 1.0):
            i-=1
            break


def layer(merged_data):
    global unique_list, old_extended_gt, new_extended_gt
    temp_data = merged_data

    # finiding initial transitivity in dataframe
    df_transative = pd.DataFrame(columns=['left_spec_id', 'right_spec_id'])
    for i in unique_list:
        j = temp_data.loc[temp_data['left_spec_id'] == i]
        k = temp_data.loc[temp_data['right_spec_id'] == i]
        t = []
        t = t + list(j['right_spec_id'])
        t = t + list(k['left_spec_id'])
        for b in t:
            a = temp_data.loc[temp_data['left_spec_id'] == b]
            c = temp_data.loc[temp_data['right_spec_id'] == b]
            d = []
            d = d + list(a['right_spec_id'])
            d = d + list(c['left_spec_id'])
            e = t + list((set(d) - set(t)))
        t = e
        df_transative = df_transative.append({'left_spec_id': i, 'right_spec_id': t}, ignore_index=True)

    # expanding the initial transitive data in the form of dataframe
    df_Column = ['left_spec_id', 'right_spec_id']
    progressive_id = 0
    progressive_id2row_df = {}
    match_pairs = df_transative
    # match_pairs['left_spec_id'] = match_pairs['right_spec_id'].apply(literal_eval)
    match_pairs_dict = match_pairs.set_index('left_spec_id')['right_spec_id'].to_dict()
    for key in match_pairs_dict.keys():
        for value in match_pairs_dict[key]:
            row = (key, value)
            progressive_id2row_df.update({progressive_id: row})
            progressive_id += 1
    initial_key_value_pair = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=df_Column)

    # Removing Duplicates
    removing_duplicates_df1 = initial_key_value_pair
    removing_duplicates_df1['label'] = 0
    removing_duplicates_df1['label'] = np.where(
        (removing_duplicates_df1['left_spec_id'] == removing_duplicates_df1['right_spec_id']), 1,
        removing_duplicates_df1['label'])
    removing_duplicates_df1 = removing_duplicates_df1[removing_duplicates_df1.label != 1]
    removing_duplicates_df2 = removing_duplicates_df1.drop(['label'], axis=1)

    matched_pair_df3 = pd.DataFrame(np.sort(removing_duplicates_df2[['left_spec_id', 'right_spec_id']], axis=1))

    matched_pair_df4 = matched_pair_df3.drop_duplicates()

    extended_df = matched_pair_df4.rename(columns={0: "left_spec_id", 1: 'right_spec_id'})
    #print(extended_df)
    return extended_df


if __name__ == "__main__":
    main()