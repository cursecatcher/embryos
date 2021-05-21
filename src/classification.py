#!/usr/bin/env python3 

import argparse
import numpy as np 
import os 
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt 
from multiprocessing import Pool

import sklearn.metrics as metrics 

import data_preparation as dp 
import pipelines as pp  
import features as fs 
import utils 


class PipelinesEvaluator:
    def __init__(self, X, y, n_folds, target_labels):
        self.X = X
        self.y = y 
        self.__n_folds = n_folds
        self.__targets = target_labels
        self.__evaluations = list() 
        self.__avg_rocs = dict() 


    def plot_averaged_roc(self, output_folder):
        mean_fpr = np.linspace(0, 1, 100)

        for pipeline, tprs in self.__avg_rocs.items():
            # aucs = [metrics.auc(mean_fpr, tpr) for tpr in tprs]
            # std_auc = np.std(aucs, axis=0)

            mean_tpr = np.mean(tprs, axis=0).ravel()
            std_tpr = np.std(tprs, axis=0).ravel()

            tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
            mean_fpr = np.linspace(0, 1, 100)

            mean_auc = metrics.auc(mean_fpr, mean_tpr)

            fig, ax = plt.subplots()

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

            ax.plot(mean_fpr, mean_tpr, color="b", 
                label=r'AUC = {:.2f}'.format(mean_auc),# $\pm$ {:.2f}'.format(mean_auc, std_auc),
                lw=2, alpha=.8)

            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                            alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], title="")

            ax.legend(loc="lower right")
            ax.set_xlabel("1-Specificity")
            ax.set_ylabel("Sensitivity")
            output_filename = os.path.join(output_folder, f"ROC_{pipeline}.pdf")
            plt.savefig(fname=output_filename, format="pdf")
            plt.close(fig)


            print(f"{pipeline}: AUC = {mean_auc:.5f}")
    
    
    def __iter__(self):
        for ev in self.__evaluations:
            yield ev 
    
    def __len__(self):
        return len(self.__evaluations)

    def evaluate(self, pipelines, output_folder, filename_prefix=""):
        tester = fs.PipelineEvaluator(pipelines, self.__targets, output_folder)
        samples_report, rocs_info = tester.test_cv(self.X, self.y, self.__n_folds, filename_prefix)
        samples_report.to_csv(os.path.join(output_folder, "samples_report.tsv"), sep="\t")
        tester.visualize(filename_prefix)
        metrics_report = tester.metrics() 
        metrics_report.to_csv(
            os.path.join(output_folder, "classification_report.tsv"), 
            sep="\t", float_format="%.3g"
        )
        self.__evaluations.append(tester)

        for pipeline, rocs_data in rocs_info.items(): 
            pipeline_name = fs.PipelineEvaluator.get_pipeline_name(pipeline)
            try:
                self.__avg_rocs[pipeline_name].append(rocs_data)
            except KeyError:
                self.__avg_rocs[pipeline_name] = [rocs_data]

        return metrics_report 



def evaluate_pipelines(evaluator, global_output_folder, base_output_folder):
    pipeline_methods = [
        pp.EstimatorWithoutFS,
        # pp.KBestEstimator,
        # pp.FromModelEstimator
    ]
    tmp = sum([method(evaluator.X).get_pipelines() for method in pipeline_methods], [])
    pipelines, _ = zip(*tmp)

    return evaluator.evaluate(
        pipelines, 
        utils.make_folder(global_output_folder, base_output_folder)
    )

def replicate_experiment(dataset, output_folder, dataset_name, n_replicates, n_folds, validation_set = None):
    session_folder = os.path.join(dataset_name, "runs", "run_{}")
    evaluator = PipelinesEvaluator(dataset.data, dataset.target, n_folds, dataset.target_labels)
    dataframes = list() 

    print(f"Running experiment {n_replicates} times")

    for n in range(n_replicates):
     #   print(".", end="", flush=True)  
        dataframes.append(
            evaluate_pipelines(
                evaluator, output_folder, session_folder.format(n+1))
        )
    
    evaluator.plot_averaged_roc(os.path.join(output_folder, dataset_name))

    return dataframes


def classification_task(dataset, mirna_list, output_folder, n_replicates, n_folds, validation_set = None):
    print(f"\nmiRNA list: {mirna_list}")
    mirna_list_name = os.path.basename(os.path.splitext(mirna_list)[0])

    data = dataset.select_features(feature_file = mirna_list) 
    validation = validation_set.select_features(feature_file = mirna_list) if validation_set is not None else None 
    

    report_name = f"classification_report_{mirna_list_name}.tsv"
    replicates = replicate_experiment(
        data, 
        output_folder, 
        mirna_list_name, 
        n_replicates=n_replicates, 
        n_folds=n_folds, 
        validation_set = validation)
    #
    reduce(
        lambda x, y: x.add(y, fill_value=0), replicates)\
            .apply(lambda row: row / n_replicates, axis=1)\
            .to_csv(
                os.path.join(output_folder, mirna_list_name, report_name), 
                sep="\t", 
                float_format="%.3g"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--output", dest="output_folder", type=str)
    #
    parser.add_argument("-i", "--input", dest="input_dataset", type=str, required=True)
    #explicit dataset used only for validation purposes
    parser.add_argument("-v", "--validation", dest="validation_set", type=str)
    # target covariate to predict 
    parser.add_argument("-t", "--target", dest="target_covariate", type=str, required=True)
    # mirna list to restrict the count matrix 
    parser.add_argument("-f", dest="feature_lists", nargs="*", default=list())
    # covariates to ignore 
    parser.add_argument("--ignore", dest="covariates_to_ignore", type=str, nargs="*")
    # list of covariates to use 
    parser.add_argument("--use_this", dest="covariates_to_use", type=str, nargs="*")
    # binary classification target 
    parser.add_argument("--binary", dest="binary_classification_targets", type=str, nargs=2)
    # number of replicates 
    parser.add_argument("--nrep", dest="num_replicates", type=int, default=1)
    # number of folds of K fold CV 
    parser.add_argument("--folds", dest="num_folds", type=int, default=3)

    args = parser.parse_args() 

    filename = args.input_dataset 
    output_path = args.output_folder 
    covs2ignore = args.covariates_to_ignore
    covs2use = args.covariates_to_use 
    target_labels = args.binary_classification_targets

    datasets = dp.Dataset.load_input_data(
        filename, args.target_covariate, target_labels, 
        covs2ignore = covs2ignore, covs2use = covs2use
    )
    dataset = datasets[list(datasets.keys())[0]]


    validation = None 
    if args.validation_set:
        validation = dp.Dataset.load_input_data(
            args.validation_set, args.target_covariate, target_labels, 
            covs2ignore = covs2ignore, covs2use = covs2use) 
#            index_column = args.index_column)
        #get Dataset from dict of Dataset 
        name = list(validation.keys())[0]
        validation = validation[name]

        print(f"Validation set loaded: {name}")



    feature_lists = list() 
    for feature_list in args.feature_lists:
        if os.path.isfile(feature_list):
            feature_lists.append(feature_list)
        elif os.path.isdir(feature_list):
            for root, folders, files in os.walk(feature_list):
                feature_list.extend([os.path.join(root, f) for f in files])

    
    for chosen_features in feature_lists:
        print(f"Running classification task using {chosen_features} file")
        
        classification_task(
            dataset, chosen_features, args.output_folder,
            n_replicates=args.num_replicates, n_folds=args.num_folds, 
            validation_set=validation) 


    # dataset = dp.Dataset.do_preprocessing(
    #     args.count_matrices, args.covariates, args.covariate_types, 
    #     args.target_covariate, args.binary_classification_targets,
    #     args.covariates_to_use, args.covariates_to_ignore
    # )
    

    #to parallelize ? 

    #do experiments, then merge the results in a single dataframe 
    # for mirna_list in args.mirna_list:
    #     if os.path.isfile(mirna_list):
    #         classification_task(dataset, mirna_list, args.output_folder, n_replicates=args.num_replicates, n_folds=args.num_folds)
    #     elif os.path.isdir(mirna_list):
    #         print("Entering in {} directory".format(mirna_list))
    #         for content in os.listdir(mirna_list):
    #             classification_task(dataset, os.path.join(mirna_list, content), args.output_folder, n_replicates=args.num_replicates, n_folds=args.num_folds)


