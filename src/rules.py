#!/usr/bin/env python3 

import argparse
import os, six, sys
sys.modules["sklearn.externals.six"] = six 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from collections import defaultdict, Counter
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from skrules import DualSkoper, Rule, RuleModel

import data_preparation as dp 
from sampling import smote, bordersmote, adasyn


def get_train_val_data(dataset: dp.Dataset, feature_list_filename: str, validation = None):
    #extract subset of features from dataset
    curr_dataset = dataset.select_features(feature_list_filename)
    #split in training/test and validation sets
    if validation:
        print("Validation set loaded from external source")
        #fake split 
        validation = validation.select_features(feature_list_filename)

        df_train, y_train = curr_dataset.data, curr_dataset.target
        df_val, y_val = validation.data, validation.target 
    else:
        print("Validation set extracted from training set")
        #split dataset in two portions 
        X_train, X_val, y_train, y_val = train_test_split(
            curr_dataset.data, 
            curr_dataset.target, 
            test_size=0.1,
            stratify=curr_dataset.target)

        #rebuild dataframes from numpy matrices 
        df_train = pd.DataFrame(X_train, columns = curr_dataset.data.columns)
        df_val = pd.DataFrame(X_val, columns = curr_dataset.data.columns)

    return df_train, df_val, y_train, y_val



def enrich_dataframe(df):
    stats = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    df_stats = df.describe().loc[stats]
    return pd.concat([df, df_stats])

def reformat_ruleset(ruleset):
    dictionary = { rule: stats for (rule, stats) in ruleset }
    return pd.DataFrame.from_dict(dictionary).T


def save_sheet(df, writer, sheet_name):
    try:
        df.to_excel(writer, sheet_name=sheet_name, float_format="%.3f", index=True)
    except:
        print(f"Cannot write {sheet_name} sheet")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #input dataset used for training (and eventually validation)
    parser.add_argument("-i", "--input", dest="input_db", type=str, required=True)
    #explicit dataset used only for validation purposes
    parser.add_argument("-v", "--validation", dest="validation_set", type=str)
    #output folder for analysis' results
    parser.add_argument("-o", "--output", dest="output_path", type=str, required=True)
    #target feature to predict 
    parser.add_argument("-t", "--target", dest="target_feature", type=str, required=True)
    #
    parser.add_argument("-l", "--labels", dest="labels_to_predict", type=str, nargs=2, required=True)
    #provide a set of feature lists to reduce dataset dimensionality 
    parser.add_argument("-f", dest="feature_lists", nargs="*", default=list())
    parser.add_argument("-r", "--num_rep", dest="num_replicates", type=int, default=1)
    parser.add_argument("--ignore", dest="features_to_ignore", nargs="*", default=list())
    parser.add_argument("--n_cv", dest="num_folds_cv", default=3, type=int)
    parser.add_argument("--index", dest="index_column", type=str)
    
    args = parser.parse_args()

    filename = args.input_db
    output_path = args.output_path
    covariates_to_ignore = args.features_to_ignore
    covariate_to_predict = args.target_feature
    target_labels = args.labels_to_predict

    # datasets = dp.Dataset.read_from_excel(
    #     filename, covariate_to_predict, target_labels, 
    #     covs2ignore = covariates_to_ignore, 
    #     index_column = args.index_column
    # ) 

    datasets = dp.Dataset.load_input_data(
        filename, covariate_to_predict, target_labels, 
        covs2ignore = covariates_to_ignore, 
        index_column = args.index_column)

    validation = None 
    if args.validation_set:
        validation = dp.Dataset.load_input_data(
            args.validation_set, covariate_to_predict, target_labels, 
            covs2ignore = covariates_to_ignore, 
            index_column = args.index_column)
        validation = validation[args.validation_set]


    if len(args.feature_lists) == 0:
        exit("No feature list provided. Addios.")
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    feature_lists = list() 
    for feature_list in args.feature_lists:
        if os.path.isfile(feature_list):
            feature_lists.append(feature_list)
        else:
            for root, folders, files in os.walk(feature_list):
                feature_lists.extend([os.path.join(root, f) for f in files])


    for feature_file in feature_lists:
        print(f"Using feature file: {feature_file}")
        
        result_per_dataset = dict()

        for name, dataset in datasets.items():
            print(f"Current dataset: {name}")
            print(f"Class proportion: {Counter(dataset.target)}")

            df_train, df_val, y_train, y_val = get_train_val_data(
                dataset, feature_file, validation)

            print(df_train.describe())
            print(df_val.describe())

            print(f"Data proportion in (original) training-test set: {Counter(y_train)}")
            print(f"Data proportion in validation set: {Counter(y_val)}")

            ##train model based on recall
            print("Training model based on recall", flush=True)
            clf_params = dict(
                n_cv_folds = args.num_folds_cv, 
                n_repeats = args.num_replicates, 
                n_estimators = 500
            )

            rm_recall = RuleModel(recall_min=0.7, **clf_params)
            rm_recall.fit(df_train, y_train)

            ##train model based on precision 
            print("Training model based on precision", flush=True)
            rm_precision = RuleModel(precision_min=0.7, **clf_params)
            rm_precision.fit(df_train, y_train)
            
            ##merge the two rulesets previously obtained
            rm = RuleModel(recall_min=0.7, precision_min=0.7)
            rm.ruleset = rm_recall.ruleset + rm_precision.ruleset
            
            ##validate the ruleset against the validation set 
            rm.validate(df_val, y_val)

            print("Rules survived to the validation:")

            for x, _ in rm.ruleset:
                print(x)

            print(flush=True)

            result_per_dataset[name] = dict(
                ruleset = rm.ruleset, 
                reports = rm.reports, 
                test_whole = rm.test_rules(dataset.data, dataset.target),
                test_training = rm.test_rules(df_train, y_train),
                test_valid = rm.test_rules(df_val, y_val)
            )


        
        feature_list_name = "_".join(os.path.basename(feature_file).split(".")[:-1])
        fln = f"{feature_list_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        xlsx_file = f"{output_path}/{fln}.xlsx"

        with pd.ExcelWriter(xlsx_file, mode="w") as writer:
            print(f"Saving results in {xlsx_file}")
            
            for dataset_name, data2save in result_per_dataset.items():
                for key in ("ruleset", "test_whole", "test_training", "test_valid"):
                    data = data2save.get(key)
                    df = reformat_ruleset(data)

                    save_sheet(df, writer, key) 
