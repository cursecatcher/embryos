#!/usr/bin/env python3 

import argparse
import pandas as pd 
import os 
from collections import defaultdict
from functools import reduce 
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import data_preparation as dp 
import classification as clfy 
from kbest import BestFeatureNumberFinder
import utils



def classification(dataset, feature_list, output_folder, n_replicates, n_folds, sheet_name):
    subdataset = dataset.select_features(feature_list)
    feature_list_name = os.path.splitext(os.path.basename(feature_list))[0]

    outfoldername = f"embryo_{sheet_name}"
    dataframes = clfy.replicate_experiment(
        subdataset, 
        output_folder, 
        os.path.join(outfoldername, feature_list_name), 
        n_replicates=n_replicates, 
        n_folds=n_folds
    ) 
    report_name = f"classification_report_{sheet_name}_{feature_list_name}.tsv"
    reduce(lambda x, y: x.add(y, fill_value=0), dataframes)\
        .apply(lambda row: row / len(dataframes), axis=1)\
            .to_csv(
                os.path.join(output_folder, outfoldername, report_name), 
                sep="\t", 
                float_format="%.3g"
            )

def kbest(dataset, sheet, output_folder, num_trials):
    print(f"Feature selection over {sheet} sheet...")

    current_output_folder = os.path.join(output_folder, f"embryo_{sheet}", "fselection")
    kfinder = BestFeatureNumberFinder(current_output_folder, dataset.target_labels)
    kfinder.evaluate(dataset, num_trials=num_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_db", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output_folder", type=str, required=True)
    parser.add_argument("-t", "--target" , dest="target_feature", type=str, required=True)
    parser.add_argument("-f", dest="feature_lists", nargs="*", default=list())
    parser.add_argument("-r", "--num_rep", dest="num_replicates", type=int, default=0)
    parser.add_argument("--ignore", dest="features_to_ignore", nargs="*", default=list())
    parser.add_argument("--mining", dest="enable_mining", action="store_true")
    args = parser.parse_args()

    num_replicates = args.num_replicates if args.num_replicates > 0 else 100 
    filename = args.input_db
    output_folder = args.output_folder
    covariates_to_ignore = [
        "Patient ID"
    ]
    covariates_to_ignore.extend(args.features_to_ignore)

    covariate_to_predict = args.target_feature
    target_labels = ["NO", "YES"]

    datasets = dp.Dataset.read_from_excel(
        filename, covariate_to_predict, target_labels, 
        covs2ignore = covariates_to_ignore
    )


    if args.feature_lists:
        print("Rule mining ")

        if args.enable_mining:
            raise NotImplementedError()
                        
        else:
            print("Classification using the specified features...")

            for feature_list in args.feature_lists:
                for sheet_name, dataset in datasets.items():
                    classification(dataset, feature_list, output_folder, num_replicates, 10, sheet_name)

    else: 
        #feature selection d
        for sheet, dataset in datasets.items():
            print(f"Feature selection on {sheet} sheet")
            kbest(dataset, sheet, output_folder, num_replicates)

