#!/usr/bin/env python3 

import argparse
import os, six, sys
sys.modules["sklearn.externals.six"] = six 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, cohen_kappa_score

from skrules import SkopeRules
from rulefit import RuleFit


import data_preparation as dp 


# def apply_skrules(dataset, folds):
#     clf = SkopeRules(feature_names=dataset.data.columns, n_estimators=333)
#     X, y = dataset.data, dataset.target
    
#     for idx_train, idx_test in folds: 
#         X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
#         y_train, y_test = y[idx_train], y[idx_test]

#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)

#         print(f"Baseline:\n{classification_report(y_test, y_pred)}")

#         for nrules in range(1, 15):
#             y_pred = clf.predict_top_rules(X_test, nrules)
#             print(f"Performance using {nrules} rules:\n{classification_report(y_test, y_pred)}\n")


# def apply_rulefit(dataset, folds):
#     X, y = dataset.data, dataset.target

#     for n, (idx_train, idx_test) in enumerate(folds, 1):
#         X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
#         y_train, y_test = y[idx_train], y[idx_test]

#         clf = RuleFit(rfmode="classify").fit(X_train.values, y_train, feature_names=dataset.data.columns)
#         y_pred = clf.predict(X_test.values)

#         print(f"Performance on {n}th fold:\n{classification_report(y_test, y_pred)}")


#         print(f"Importances:\n{clf.get_feature_importance()}")



def stuff(label, stats):
    return {
        f"{label}_{stat}": value for stat, value in stats.items()
    } 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_db", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output_folder", type=str, required=True)
    parser.add_argument("-f", dest="feature_lists", nargs="*", default=list())
    parser.add_argument("-r", "--num_rep", dest="num_replicates", type=int, default=0)
    parser.add_argument("--ignore", dest="features_to_ignore", nargs="*", default=list())
    parser.add_argument("--mining", dest="enable_mining", action="store_true")
    args = parser.parse_args()

    filename = args.input_db
    output_folder = args.output_folder
    covariates_to_ignore = [
        "Tecnica", ##use this 
        "ID Paziente",
        "tPNa", 
        "tPNf-tPNa"
    ]
    covariates_to_ignore.extend(args.features_to_ignore)

    covariate_to_predict = "Output"
    target_labels = ["NO", "SI"]

    datasets = dp.Dataset.read_from_excel(
        filename, None, covariate_to_predict, target_labels, 
        covs2ignore = covariates_to_ignore
    )

    n_splits = 5 
    n_estimators = 100 
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    classifiers = {
        # "rulefit_gb": (RuleFit, dict(
        #     rfmode = "classify", 
        #     tree_generator = GradientBoostingClassifier(n_estimators=n_estimators), 
        #     n_jobs = -1
        # )), 
        # "rulefit_rf": (RuleFit, dict(
        #     rfmode = "classify", 
        #     tree_generator = RandomForestClassifier(n_estimators=n_estimators), 
        #     n_jobs = -1
        # )), 
        "skoperules": (SkopeRules, dict(
    #        feature_names = dataset.data.columns, 
            n_estimators = n_estimators, 
            n_jobs = -1, 
            precision_min = 0.6, recall_min = 0.5
        ))
    }


    result_per_dataset = dict()


    for name, dataset in datasets.items():
        print(f"Current dataset:\t{name}")
        folds = list(kfold.split(dataset.data, dataset.target))

        result_per_dataset[name] = performances = {clf: list() for clf in classifiers.keys()}

        for n, (idx_train, idx_test) in enumerate(folds, 1):
            print(f"\nTraining on fold #{n}...", end="", flush=True)

            X_train, X_test = dataset.data.iloc[idx_train], dataset.data.iloc[idx_test]
            y_train, y_test = dataset.target[idx_train], dataset.target[idx_test]


            for clf_name, (clf_class, clf_args) in classifiers.items(): 
                print(f"{clf_name}...", end="", flush=True)
                #fit classifier on current training set, then predict on the relative test set 
                
                if clf_name == "skoperules":
                    clf = clone(clf_class(**clf_args, feature_names=dataset.data.columns)).fit(X_train.values, y_train)
                #for skoperules, after the classifier has been fitted, labels are predicted using the 1...N top rules
        #            print("Training skope rules classifier...")
                    for n_rules in range(1, 20):
                        prediction = clf.predict_top_rules(X_test, n_rules)
                        report = classification_report(y_test, prediction, output_dict=True)
                        class_0, class_1 = [stuff(c, report.get(c)) for c in ("0", "1")]
                        performances[clf_name].append(dict(
                            n_rules = n_rules,
                            n_fold = n,
                            **class_0, 
                            **class_1, 
                            accuracy = report.get("accuracy"),
                            cohen_k = cohen_kappa_score(y_test, prediction), 
                            rule = clf.rules_[n_rules - 1][0]
                        ))
                else:
     #               print("Training RuleFit")
                    clf = clone(clf_class(**clf_args)).fit(X_train.values, y_train)
                    #for rulefit, the classifier is trained and used in the normal way 
                    prediction = clf.predict(X_test.values)
                    #get classification stats
                    report = classification_report(
                        y_test, 
                        prediction, 
                        output_dict=True)
                    class_0 = stuff("0", report.get("0"))
                    class_1 = stuff("1", report.get("1"))
                    performances[clf_name].append(dict(
                        n_fold = n,
                        **class_0, 
                        **class_1, 
                        accuracy = report.get("accuracy"),
                        cohen_k = cohen_kappa_score(y_test, prediction)
                    ))
    

    with pd.ExcelWriter(f"{output_folder}/embrioni.xlsx", mode="w") as writer:
        for dataset_name, perf_on_dataset in result_per_dataset.items():
            for clf, results in perf_on_dataset.items():
                pd.DataFrame.from_dict(results).to_excel(
                    writer, 
                    sheet_name=f"{dataset_name} {clf}", 
                    float_format="%.3f", 
                    index=False
                )


