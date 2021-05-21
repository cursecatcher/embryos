#!/usr/bin/env python3 

import argparse
import os, six, sys
sys.modules["sklearn.externals.six"] = six 
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
from datetime import datetime
from itertools import chain

from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from skrules import RuleModel

import data_preparation as dp 
# from sampling import smote, bordersmote, adasyn


def get_train_val_data(dataset: dp.Dataset, feature_list_filename: str, validation = None):
    #extract subset of features from dataset
    curr_dataset = dataset.select_features(feature_list_filename)

    #split in training/test and validation sets
    if validation:
        print("Validation set loaded from external source")
        validation = validation.select_features(feature_list_filename)

        # df_train, y_train = curr_dataset.data, curr_dataset.target
        # df_val, y_val = validation.data, validation.target 

        training_data = curr_dataset 
        validation_data = validation 

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

        training_data = dp.Dataset(X=df_train, y=y_train)
        validation_data = dp.Dataset(X=df_val, y=y_val)

    print(f"Training variables: {training_data.data.columns}")
    print(f"Validation variables: {validation_data.data.columns}")
    
    return training_data, validation_data


def build_rule_model(dataset: dp.Dataset, validation: dp.Dataset, **kwargs):
    threshold = 0.7
    models = (
        ("recall", RuleModel(recall_min=threshold, **kwargs)), 
        ("precision", RuleModel(precision_min=threshold, **kwargs)),
        ("rec_&_prec", RuleModel(recall_min=threshold, precision_min=threshold, **kwargs))
    )
    
    for n, m in models: 
        print(f"Training model based on {n}", flush=True)
        m.fit(dataset.data, dataset.target)

        for r in m.ruleset:
            print(r)

    model = RuleModel(recall_min=threshold, precision_min=threshold)
    model.ruleset = list(chain.from_iterable([m.ruleset for _, m in models]))
    model.validate(validation.data, validation.target)

    return model

def write_to_excel(output_folder, feature_file, results_dict):
    feature_list_name = "_".join(os.path.basename(feature_file).split(".")[:-1])
    fln = f"{feature_list_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    xlsx_file = f"{output_folder}/{fln}.xlsx" 

    with pd.ExcelWriter(xlsx_file, mode="w") as writer: 
        print(f"Writing results in {xlsx_file}")

        for key, table in results_dict.items():
            save_sheet(table, writer, key)

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

    num_replicas = 10

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
        #get Dataset from dict of Dataset 
        name = list(validation.keys())[0]
        validation = validation[name]


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
            print(f"Current dataset: {name}\nClass proportion: {Counter(dataset.target)}", flush=True)

            generator = dataset.select_features(feature_file).get_train_validation(num_rep=num_replicas)
            evaluations = dict() 
            all_rules = list() 

            for n, (t, v) in enumerate(generator, 1):
                rm = build_rule_model(
                    t, v, 
                    n_cv_folds = args.num_folds_cv, 
                    n_repeats = args.num_replicates, 
                    n_estimators = 200 
                )
                print(f"Iteration #{n}: {len(rm.ruleset)} rules kept.", flush=True)

                if rm.ruleset:
                    all_rules.extend(rm.ruleset)
                    evaluations[f"test_{n}"] = reformat_ruleset(rm.test_rules(v.data, v.target))

            evaluations["all_rules"] = pd.Series([str(rule) for rule, _ in all_rules], name="rules", dtype=str)
            print(f"Tot of rules kept: {len(evaluations['all_rules'])}", flush=True)

            write_to_excel(output_path, feature_file, evaluations)


