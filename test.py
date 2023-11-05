import gzip
import json
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import argparse

def sequence_to_features(seq):
    """
    Convert a 7-letter sequence into binary features for the first and last letters.
    
    Parameters:
    - seq: A string of 7 letters (A, T, C, G)
    
    Returns:
    - A dictionary with binary features for the first and last letters.
    """
    letters = ['A', 'T', 'C', 'G']
    features = {}
    
    first_char = seq[0]
    last_char = seq[-1]
    
    for letter in letters:
        features[f"1{letter}"] = first_char == letter
        features[f"7{letter}"] = last_char == letter
            
    return features

def get_5_mer_first_last(df):
    df['5-mer'] = df['combined nucleotides'].apply(lambda x: x[1:-1])
    df = pd.get_dummies(df, columns=['5-mer'], drop_first=False)
    feature_dicts = df['combined nucleotides'].apply(sequence_to_features)
    features_df = feature_dicts.apply(pd.Series)
    df = pd.concat([df, features_df], axis=1)
    df = df.drop(columns=['combined nucleotides'])
    
    return df

def get_summary_mean_std(df):
    count_df1 = df.groupby(['transcript_id', 'transcript_position']).size().reset_index(name='read_count')
    count_df2 = df.groupby(['transcript_id']).size().reset_index(name='expression_count') #most genes are likely to produce unique transcripts
    df = df.merge(count_df1, on=['transcript_id', 'transcript_position'])
    df = df.merge(count_df2, on=['transcript_id'])
    
    summary_df = df.groupby(['transcript_id', 'transcript_position']).agg({
        #'gene_id': 'first',
        'combined nucleotides': 'first',
        'dwelling_time1': ['mean', 'std'],
        'sd1': ['mean', 'std'],
        'mean1': ['mean', 'std'],
        'dwelling_time2': ['mean', 'std'],
        'sd2': ['mean', 'std'],
        'mean2': ['mean', 'std'],
        'dwelling_time3': ['mean', 'std'],
        'sd3': ['mean', 'std'],
        'mean3': ['mean', 'std'],
        #'label': lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()
    summary_df.columns = summary_df.columns.map('_'.join)
    summary_df = summary_df.rename(columns={'label_<lambda>': 'label',
                                           'transcript_id_': 'transcript_id',
                                           'transcript_position_': 'transcript_position',
                                           'gene_id_first': 'gene_id',
                                           'combined nucleotides_first': 'combined nucleotides'})
    
    return summary_df

def main(df,model,output):
    objs0 = []
    with gzip.open(df, 'rt') as f:
        for line in f:
            objs0.append(json.loads(line))
    rows0 = []
    for i in range(len(objs0)):
        for key1, inner_dict in objs0[i].items():
            for key2, inner_list in inner_dict.items():
                for key3, sublist in inner_list.items():
                    for subsublist in sublist:
                        rows0.append([key1, key2, key3] + subsublist)
                        
    df0 = pd.DataFrame(rows0, columns=["transcript_id", "transcript_position", "combined nucleotides", "dwelling_time1", "sd1", "mean1", "dwelling_time2", "sd2", "mean2", "dwelling_time3", "sd3", "mean3"])
    df0['transcript_position'] = df0['transcript_position'].astype(int)
    summary_mean_std = get_summary_mean_std(df0)
    summary_df = summary_mean_std.copy()
    summary_mean_std = get_5_mer_first_last(summary_mean_std)
    xgb_model_1 = xgb.XGBClassifier()
    xgb_model_1.load_model(model)
    X_min_features_1 = summary_mean_std.drop(columns=['transcript_id'])
    scaler = MinMaxScaler()
    X_min_features_1_non_bool = X_min_features_1.select_dtypes(exclude=['bool'])
    X_min_features_1_bool = X_min_features_1.select_dtypes(include=['bool'])
    X_min_features_1_non_bool_scaled = scaler.fit_transform(X_min_features_1_non_bool)
    X_final = np.hstack([X_min_features_1_non_bool_scaled,X_min_features_1_bool.values])
    X_final_proba = xgb_model_1.predict_proba(X_final)[:,1]
    summary_df_scaled = summary_df.copy()
    summary_df_scaled['score'] = X_final_proba.tolist()
    summary_df_scaled[["transcript_id","transcript_position","score"]].to_csv(output,index=False)
    
    return

def get_arguments():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--model_path', required=True, help='path to the model file')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args.text_path,args.model_path,args.output_path)

#main("dataset0.json.gz","xgbmodel1.json")
#to use it in aws, remove python 3.8, install python 3.11, install 3.11 pip, install packages xgboost==2.0.0
#scikit-learn==1.2.2
#numpy==1.24.3
#pandas==2.0.3
#then type
#python3.11 test.py --text_path DATASETFILENAME --model_path MODELFILENAME --output_path OUTPUTFILENAME

