#!/usr/bin/env python3 

import argparse
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os 
import utils 
from functools import reduce 
from multiprocessing import Pool 

import data_preparation as dp 
import pipelines as pp 
import classification as clfy

#state of the art 
#faccio N run, in cui eseguo una 10fold-cv testando vari classificatori
#usando le migliori k features, con k = 1, 2, ... max features
#per ogni clf, trovo il numero di features associato alla auc massima 

#todo: 
#una volta trovato il best k per ogni clf, 
#fornire insieme di features belle "mediando" i risultati ottenuti nelle N runs


class BestFeatureNumberFinder:
    def __init__(self, output_folder, target_labels):
        self.__output_folder = output_folder
        self.__target_labels = target_labels

        #just a string to indicate the number of features' column name 
        self.__n_features_column_name = "n_features" 
        #a list of length equal to the maximum number of features
        #the k-th element of the list is another list of length equal to the number of evaluation trial
        #which contains the result of the evaluations using k features 
        self.__evaluation_k_features = list()

    
    def evaluate(self, dataset, num_trials):
        tmp_folder = os.path.join(self.__output_folder, "replicates", "rep_{}")
        # train N times the classifiers on (X,y) dataset saving the 
        # results in different folders named as rep_{n_iter}
        X, y = dataset.data, dataset.target 
        dfs = list() 
        for n in range(num_trials):
            print("\nReplicate #{}".format(n + 1))
            dfs.append(self.__generate_data(X, y, tmp_folder.format(n), self.__target_labels))

        print()

        #merge the results of the previous step computing an average on them 
        final_df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)\
            .apply(lambda row: row / num_trials, axis=1)

        measures = ["auc"]

        self.__plot_trend(final_df, measures)
        self.make_report(final_df, measures)


    def __generate_data(self, X, y, output_folder, target_labels):
        df_list = list()
        n_tot_features =  X.shape[1]

        for k in range(1, n_tot_features + 1):
            print("\nNum feature = {} / {}".format(k, n_tot_features), end="")

            #instantiate pipelines 
            kbest = pp.KBestEstimator(X, k).get_pipelines()
            sfm = pp.FromModelEstimator(X, k).get_pipelines()

            pipelines = [pipeline for pipeline, _ in kbest + sfm]

            evaluator = clfy.PipelinesEvaluator(X, y, n_folds=10, target_labels=self.__target_labels)
            df_eval = evaluator.evaluate(pipelines, os.path.join(output_folder, "k_{}".format(k)))
            df_eval[self.__n_features_column_name] = k
            df_list.append(df_eval)

            try:
                #append evaluator to the k-th list 
                self.__evaluation_k_features[k-1].append(evaluator)
            except IndexError:
                #add the k-th list if it doesn't exist
                self.__evaluation_k_features.append([evaluator])
                        
            
        return pd.concat(df_list)

    def make_report(self, df, measures):
        if not isinstance(measures, list):
            measures = [measures]

        for measure in measures:
            print("Measure: {}".format(measure.upper()))
            current_outfolder = utils.make_folder(self.__output_folder, "ranking_{}".format(measure))

            for clf_name, subdf in df.groupby(df.index):
                sorted_df = subdf.sort_values(by=[measure, self.__n_features_column_name], ascending=[False, True])
                sorted_df.to_csv(
                    os.path.join(current_outfolder, "{}_x_feature.{}.tsv".format(clf_name, measure)), 
                    sep="\t", 
                    float_format="%.4g"
                )

                #da utilizzare o togliere 
                # best_k = int(sorted_df.iloc[0][self.__n_features_column_name])
                # print("{} - best number of features is {}".format(clf_name, best_k))

                for k, evaluation in enumerate(self.__evaluation_k_features, 1):
                    best_features_per_clf = list()

                    for result_run in evaluation:
                        best_features_per_clf.extend([
                            it.best_features[clf_name]["mean"].sort_values(ascending=False).index \
                                for it in result_run])

                    feature_importances = pd.DataFrame(data=[x.to_series().tolist() for x in best_features_per_clf])
                    outdir = utils.make_folder(
                        self.__output_folder, 
                        os.path.join("feature_extraction", "{}_features".format(k))
                    )
                    feature_importances.to_csv(
                        os.path.join(outdir, "report_{}.k_{}.{}.tsv".format(clf_name, k, measure)), 
                        sep="\t", 
                        header=False 
                    )



    def __plot_trend(self, df, measures):
        if not isinstance(measures, list):
            measures = [measures]
            
        for measure in measures:
            fig, ax = plt.subplots()
            
            for clf_name, subdf in df.groupby(df.index):
                ax.plot(
                    subdf[self.__n_features_column_name], subdf[measure], label=clf_name, linestyle="--", marker="o"
                )
            
            ax.set(title="{} respect to the number of features".format(measure))
            ax.legend(loc="lower right")
            #set int values on x axis
            plt.xticks(list(range(1, int(df[self.__n_features_column_name].max()) + 1)))
            plt.xlabel("Number of features")
            plt.ylabel(measure.upper())
            plt.savefig(os.path.join(self.__output_folder, "{}_plot_n_features.pdf".format(measure)))
            plt.close(fig)

def best_k_finder(dataset, mirna_list, output_folder):
    data = dataset.select_features(miRNA_file = mirna_list)
    dataset_name = os.path.basename(os.path.splitext(mirna_list)[0])

    current_output_folder = os.path.join(args.output_folder, dataset_name, "k_best")
    BestFeatureNumberFinder(current_output_folder, data.target_labels).evaluate(data, num_trials=30)    

def best_k_finder_paral(args):
    best_k_finder(*args)


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

    args = parser.parse_args() 

    dataset = dp.Dataset.do_preprocessing(
        args.count_matrices, args.covariates, args.covariate_types, 
        args.target_covariate, args.binary_classification_targets,
        args.covariates_to_use, args.covariates_to_ignore
    )

    # with Pool(os.cpu_count() - 1) as pool:
    #     pool.map(
    #         best_k_finder_paral, 
    #         [(dataset, mirna_list, args.output_folder) for mirna_list in args.mirna_list]
    #     )

    for mirna_list in args.mirna_list:
        best_k_finder(dataset, mirna_list, args.output_folder)
