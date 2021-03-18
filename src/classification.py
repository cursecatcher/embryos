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
            output_filename = os.path.join(output_folder, "ROC_{}.pdf".format(pipeline))
            plt.savefig(fname=output_filename, format="pdf")
            plt.close(fig)


            print("{}: AUC = {:.5f}".format(pipeline, mean_auc))
    
    
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

def replicate_experiment(dataset, output_folder, dataset_name, n_replicates, n_folds):
    session_folder = os.path.join(dataset_name, "runs", "run_{}")
    evaluator = PipelinesEvaluator(dataset.data, dataset.target, n_folds, dataset.target_labels)
    dataframes = list() 

    print("Running experiment {} times".format(n_replicates))

    for n in range(n_replicates):
        print(".", end="", flush=True)
        
        dataframes.append(
            evaluate_pipelines(
                evaluator, output_folder, session_folder.format(n+1))
        )
    
    evaluator.plot_averaged_roc(os.path.join(output_folder, dataset_name))

    return dataframes


def classification_task(dataset, mirna_list, output_folder, n_replicates, n_folds):
    print("\nmiRNA list: {}".format(mirna_list))
    mirna_list_name = os.path.basename(os.path.splitext(mirna_list)[0])

    data = dataset.select_features(miRNA_file = mirna_list) #dataset.select_miRNAs(mirna_list)

    report_name = "classification_report_{}.tsv".format(mirna_list_name)
    reduce(
        lambda x, y: x.add(y, fill_value=0), replicate_experiment(
            data, 
            output_folder, 
            mirna_list_name, 
            n_replicates=n_replicates, 
            n_folds=n_folds)
        ).apply(lambda row: row / n_replicates, axis=1
            ).to_csv(
                os.path.join(output_folder, mirna_list_name, report_name), 
                sep="\t", 
                float_format="%.3g"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--output", dest="output_folder", type=str)
    # count matrix
    parser.add_argument("--cm", dest="count_matrices", nargs="*", default=list())
    # covariate matrix 
    parser.add_argument("--covs", dest="covariates", nargs="*", default=list())
    # covariate info 
    parser.add_argument("--info", dest="covariate_types")
    # target covariate to predict 
    parser.add_argument("-t", "--target", dest="target_covariate", type=str, required=True)
    # mirna list to restrict the count matrix 
    parser.add_argument("-m", "--mirnas", dest="mirna_list", nargs="*", default=list())
    # covariates to ignore 
    parser.add_argument("--ignore", dest="covariates_to_ignore", type=str, nargs="*")
    # list of covariates to use 
    parser.add_argument("--use_this", dest="covariates_to_use", type=str, nargs="*")
    # binary classification target 
    parser.add_argument("--binary", dest="binary_classification_targets", type=str, nargs=2)
    # number of replicates 
    parser.add_argument("--nrep", dest="num_replicates", type=int, default=100)
    # number of folds of K fold CV 
    parser.add_argument("--folds", dest="num_folds", type=int, default=10)

    args = parser.parse_args() 

    dataset = dp.Dataset.do_preprocessing(
        args.count_matrices, args.covariates, args.covariate_types, 
        args.target_covariate, args.binary_classification_targets,
        args.covariates_to_use, args.covariates_to_ignore
    )
    

    #to parallelize ? 

    #do experiments, then merge the results in a single dataframe 
    for mirna_list in args.mirna_list:
        if os.path.isfile(mirna_list):
            classification_task(dataset, mirna_list, args.output_folder, n_replicates=args.num_replicates, n_folds=args.num_folds)
        elif os.path.isdir(mirna_list):
            print("Entering in {} directory".format(mirna_list))
            for content in os.listdir(mirna_list):
                classification_task(dataset, os.path.join(mirna_list, content), args.output_folder, n_replicates=args.num_replicates, n_folds=args.num_folds)


