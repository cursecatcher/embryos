#!/usr/bin/env python3 

import argparse
import os
import pandas as pd
from pandas.core.algorithms import mode
from skrules.rule_model import RuleModel 
import data_preparation as dp
from rules import reformat_ruleset 


def evaluate_dataset(model: RuleModel, dataset: dp.Dataset, eval_splits: bool = True):
    ev = reformat_ruleset(model.test_rules(dataset.data, dataset.target, eval_splits))
    ev = ev[(ev.accuracy > 0.6)].sort_values(by=[("f1_score", "accuracy")], ascending=False)
    # ev = ev.sort_values(by=["f1_score"], ascending=False)
    return ev 

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-i", "--filename", type=str, required=True, nargs="+") #validation set 
    #può essere sia una cartella che un file excel - XXX se esiste già, aggiungere fogli XXX
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-r", "--rules_file", type=str, required=True) #1+list of rules

    parser.add_argument("-t", "--target", dest="target_feature", type=str, required=True)
    #
    parser.add_argument("-l", "--labels", dest="labels_to_predict", type=str, nargs=2, required=True)
    args = parser.parse_args()

    filenames = args.filename
    output_path = args.output_path
    covariate_to_predict = args.target_feature
    target_labels = args.labels_to_predict


    #obtain ruleset from rules file 
    with pd.ExcelFile(args.rules_file, engine="openpyxl") as xlsx:
        df = pd.read_excel(xlsx, sheet_name="all_rules", index_col=0)
        all_rules = {rule: list() for rule in df["rules"]}

        sheets = [sheet for sheet in xlsx.sheet_names if sheet.startswith("test_")]

        for sheet in sheets:
            ruleset = pd.read_excel(xlsx, sheet_name=sheet, index_col=0).to_dict().get("tag")

            for rule, tag in ruleset.items():
                if rule in all_rules: 
                    all_rules[rule].append(tag)


    model = RuleModel.load_rules(list(all_rules.items()))
    #obtain list of used features from rules
    features = set()
    for rule, _ in model.ruleset:
        features.update({f for f, _ in rule.clauses})

    with pd.ExcelWriter(f"{args.output_path}.xlsx", engine="openpyxl") as writer:
        for f in filenames:
            for name, dataset in dp.Dataset.load_input_data(f, covariate_to_predict, target_labels, covs2use = features).items():
                cohort = f.split("/")[-1].split(".")[0].split("Cohort-")[-1]
                print(f"Evaluating {cohort} cohort against {args.rules_file} ruleset")

                try:
                    evaluate_dataset(model, dataset).to_excel(
                        writer, sheet_name=cohort, float_format="%.3f", index=True)
                except:
                    print("Some error occurred during dataset evaluation :(")
