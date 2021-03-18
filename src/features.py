from collections import Counter
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import os 
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics 

import pipelines as pp 
import utils 

matplotlib.rcParams.update({'figure.autolayout': True})

class FeatureSelector:
    def __init__(self, trained_classifier, feature_index):
        self.__pipeline = trained_classifier
        self.__initial_features = feature_index.to_series()
        self.__selected_features = feature_index.to_series()
        
    
    def remove_features_by_variance(self):
        try:
            selector_by_variance = self.__pipeline["var_threshold"]
            self.__selected_features = self.__selected_features[selector_by_variance.get_support()]
        except KeyError:
            pass 

    def get_selected_features(self):
        """ Obtain selected features from pipeline's feature selector, if it present""" 

        self.remove_features_by_variance()

        try:
            f_selector = self.__pipeline["selector"]
            self.__selected_features = self.__selected_features.loc[f_selector.get_support()]
        except KeyError:
            pass 
        
        return self.__selected_features

    def get_classifier_features(self):
        """ Obtain feature importances from classifier """

        estimator = self.__pipeline[-1]
        importances = None

        if hasattr(estimator, "coef_"):
            coefficients = estimator.coef_[0]
            importances = self.f_importance(coefficients)

        elif hasattr(estimator, "feature_importances_"):
            feature_importances = estimator.feature_importances_
            importances = self.f_importance(feature_importances)
    
        return importances
                

    def f_importance(self, coef): 
        imp, names = zip(*sorted(zip(np.abs(coef), self.__selected_features), key=lambda pair: pair[1]))
        return pd.Series(data=imp, index=names)



class PipelineEvaluator:
    def __init__(self, clf_pipelines, target_labels, output_folder):
        self.__pipelines = clf_pipelines
        self.__predictions = dict() 
        self.__features = dict()
        self.__rocs = dict() 
        self.__true_y = list()
        self.__target_labels = target_labels
        self.__output_folder = output_folder
        #attribute to get the averaged ROC over multiple runs 
        self.__avg_roc = dict() 

        self.__init()
    

    @property
    def output_folder(self):
        return self.__output_folder
    
    @output_folder.setter
    def output_folder(self, outf):
        self.__output_folder = outf 
    
    @property
    def best_features(self):
        return {
            PipelineEvaluator.get_pipeline_name(pipeline): df \
            for pipeline, df in self.__features.items()
        }
    

    def __init(self):
        for pipeline in self.__pipelines:
            ## TODO - data structure have to be replaced by a class 
            self.__predictions[pipeline] = list() 
            self.__rocs[pipeline] = None 
            self.__features[pipeline] = pd.DataFrame()

        # self.__predictions = {pipeline: list() for pipeline in self.__pipelines}
        # self.__rocs = {pipeline: None for pipeline in self.__pipelines}
        # self.__avg_roc = {pipeline: list() for pipeline in self.__pipelines}
        # self.__features = {pipeline: pd.DataFrame() for pipeline in self.__pipelines}
        # self.__true_y = list() 


    def test_cv(self, X, y, n_splits=10, file_prefix=""):
        self.__init()

        roc_plot_folder = utils.make_folder(self.__output_folder, "roc_plots")
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(X, y))
        self.__true_y = sum([list(y[idx_test]) for _, idx_test in folds], [])

        file_prefix = "ROC_{}".format(file_prefix)
        mean_fpr = np.linspace(0, 1, 100)

#        relevant_features = list()
        rules = list()

        for clf in self.__pipelines:
            tprs, aucs = list(), list() 
            clf_name = PipelineEvaluator.get_pipeline_name(clf)

            fig, ax = plt.subplots()

            for n_fold, (idx_train, idx_test) in enumerate(folds):
                X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
                y_train, y_test = y[idx_train], y[idx_test]

                self.__predictions[clf].extend(clf.fit(X_train, y_train).predict(X_test))

                try:
                    rf_rules = tr.RandomForestRuleExtractor(clf[-1])
                    rules.append(rf_rules)

                except:
                    pass 
#                    print(clf)
            



                ###
                # try:
                #     descriptor = td.ForestDescriptor(clf[-1], X.columns, self.__target_labels)
                #     relevant_features.append(descriptor.get_features(1))
                #   #  descriptor.predict(clf[-1], X_test)
                        
                # except AttributeError:
                #     pass 
                ###


                f_selector = FeatureSelector(clf, X.columns)
                _ = f_selector.get_selected_features()
                self.__features[clf]["fold_{}".format(n_fold)] = f_selector.get_classifier_features()

                viz = metrics.plot_roc_curve(
                    clf, X_test, y_test, name="", 
                    alpha=0.3, lw=1, ax=ax
                )
                interp_trp = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_trp[0] = 0.0
                tprs.append(interp_trp)
                aucs.append(viz.roc_auc)

            ###########
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color="b", 
                label=r'(AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
                lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
           
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                            alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC curve of " + clf_name)
            ax.legend(loc="lower right")

            filename = os.path.join(roc_plot_folder, "{}_{}".format(file_prefix, clf_name))
            plt.savefig(fname=filename + ".pdf", format="pdf")
            plt.close(fig)

            self.__rocs[clf] = mean_auc
            self.__avg_roc[clf] = mean_tpr

        # print("Relevant features of RF:")
        # tot = relevant_features[0]
        # for x in relevant_features[1:]:
        #     tot += x
        # print(tot)

        ##dataframe whose columns are the predictions of the tested classifiers
        ##and the rows are the tested examples
        index = sum([list(X.iloc[idxs].index) for _, idxs in folds], [])
        map_to_label = lambda x: self.__target_labels[x]

        df = pd.DataFrame(
            data = {
                PipelineEvaluator.get_pipeline_name(pipeline): list(map(map_to_label, y)) \
                    for pipeline, y in self.__predictions.items()
            }, 
            index = index #sum([list(X.iloc[idxs].index) for _, idxs in folds], [])
        )
        #counts how many times each element has been predicted to the correct class
        true_y = list(map(map_to_label, self.__true_y))
        df["right_pred"] = [list(row).count(y) for row, y in zip(df.values, true_y)]
        #add correct class column
        df["true_y"] = true_y

        return df.sort_values(by=["true_y", "right_pred"], ascending=(True, False)), self.__avg_roc

    def visualize(self, file_prefix):
        feature_folder = utils.make_folder(self.__output_folder, "feature_ranking")
        feature_plot_folder = utils.make_folder(feature_folder, "plots")

        ranking_list = list()
           
        for clf, features in self.__features.items():
            pipeline_name = PipelineEvaluator.get_pipeline_name(clf)
            #get the mean score evaluating nans as 0 
            means = features.fillna(0).mean(axis=1).sort_values()
            n_elems = len(means.index)

            plt.barh(range(n_elems), means.values, align="center")
            plt.yticks(range(n_elems), means.index)
            plt.title("Feature ranking of " + pipeline_name)

            filename = os.path.join(feature_plot_folder, "{}_{}".format(file_prefix, pipeline_name))
            plt.tight_layout()
            plt.savefig(fname=filename + ".pdf", format="pdf")
            plt.close()

            features["mean"] = means
            # save dataframe ranking features from the best to the worse (based on average score)
            features.sort_values(by="mean", ascending=False).to_csv(
                path_or_buf = os.path.join(feature_folder, pipeline_name + ".csv"), 
                sep="\t", 
                decimal=",", 
                float_format="%.3g", 
                na_rep="NA"
            )

            #obtain list of features ranked by score 
            sorted_features = list(pd.Series(data = means).sort_values(ascending=False).index)
            ranking_list.append(pd.Series(data = sorted_features, name = pipeline_name))

        #write feature ranking of each pipeline 
        pd.concat(ranking_list, axis=1, keys=[s.name for s in ranking_list]).to_csv(
            path_or_buf = os.path.join(feature_folder, "best_features_per_classifier.csv"), 
            sep="\t"
        )



    def metrics(self):
        my_data = list() 

        measures = ("precision", "recall", "f1-score", "support")
        columns = ["auc", "accuracy", "cohen-kappa", "TP", "FP", "FN", "TN"]
        init_columns_flag = True 

        for clf, predictions in self.__predictions.items():
            curr_data = list()

            report = metrics.classification_report(
                self.__true_y, predictions, 
                target_names=self.__target_labels, 
                output_dict=True)
            
            confusion_matrix = list(metrics.confusion_matrix(self.__true_y, predictions).flatten())

            curr_data = [
                self.__rocs[clf], 
                report["accuracy"], 
                metrics.cohen_kappa_score(self.__true_y, predictions), 
                *confusion_matrix
            ]
            
            for target_class in self.__target_labels:
                curr_data.extend([report[target_class][m] for m in measures])

                if init_columns_flag:
                    columns.extend(["{}_{}".format(target_class, m) for m in measures])

            my_data.append(pd.Series(curr_data, index=columns, name=PipelineEvaluator.get_pipeline_name(clf)))
            init_columns_flag = False

        return pd.concat(my_data, axis=1, keys=[s.name for s in my_data]).T
    
    @classmethod
    def get_pipeline_name(cls, pipeline):
        steps = list()

        for name, obj in pipeline[-2:].named_steps.items():
            if name == "selector":
                steps.append("kbest" if type(obj) is pp.SelectKBest else "sfm")
            else:
                steps.append(name)

        return "_".join(steps)            







