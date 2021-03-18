#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
#import seaborn as sns 
import matplotlib.pyplot as plt  

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MultipleLabelsException(Exception):
    def __init__(self, labels):
        message = (
            f"The dataset contains more than 2 labels ({labels}).\n"
            "You have to specify the --binary option.")
        super().__init__(message)


class Dataset:
    def __init__(self):
        self.__covariate_matrix = None 
        self.__target = None 
        self.__target_labels = None 

    @classmethod
    def read_from_excel(cls, excel_filename, covs_types, cov2predict, target_labels, covs2use = None, covs2ignore = None):
        datasets = dict() 

        with pd.ExcelFile(excel_filename) as xlsx:
            cov2predict = cov2predict.lower()

            for sheet_name in xlsx.sheet_names: 
                datasets[sheet_name] = dataset = Dataset() 
                df = dataset.__covariate_matrix = pd.read_excel(xlsx, sheet_name=sheet_name)
                df.columns = df.columns.str.lower() 

                bad_cols = [col for col in df.columns if col.startswith("unnamed:")]
                bad_cols.extend([col.lower() for col in covs2ignore])
                df.drop(columns=bad_cols, inplace=True)

                for colname in df.columns:
                    if colname != cov2predict:
                        df[colname] = df[colname].apply(lambda x: float(str(x).split()[0].replace(",", "")))
                        df[colname].astype("float64").dtypes

                if df.isnull().sum().sum() > 0:
                    df.fillna(df.mean(), inplace=True)
                
            # print("Plotting {} correlation matrix ".format(sheet))

            # f, ax = plt.subplots(figsize=(10, 8))
            # corr = dataset.cov_matrix.corr().abs()
        
            # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
            # plt.savefig("{}_correlation_matrix.png".format(sheet), dpi = 400)

                dataset.filter_by_target(cov2predict, target_labels)
        
        return datasets 
    
    @property
    def cov_matrix(self):
        return self.__covariate_matrix
    
    @property
    def data(self):
        return self.__covariate_matrix

    
    @property
    def target(self):
        return self.__target
    
    @property
    def target_labels(self):
        return self.__target_labels
    
    def filter_by_target(self, target_cov, allowed_values):
        """ Filter the target column removing all the values different 
        from the allowed values. 
        Remove the target covariate from the matrix and return it as output"""

        target_cov = target_cov.lower()

        if not allowed_values:
            allowed_values = set(self.cov_matrix[target_cov])
            if len(allowed_values) != 2:
                raise MultipleLabelsException(allowed_values)

        mask = self.cov_matrix[target_cov].isin(allowed_values)
        covs_matrix = self.cov_matrix[mask]
        self.__target_labels = sorted(allowed_values)
        self.__target = LabelEncoder().fit(self.__target_labels).transform(covs_matrix[target_cov])

        self.__covariate_matrix = covs_matrix.drop(columns=[target_cov])

        return self
        

    def remove_covariates(self, covariates):
        """ Remove in-place the covariates passed by argument """ 
        if covariates and self.cov_matrix:
            f2remove = set(self.cov_matrix.columns).intersection([c.lower() for c in covariates])
            if f2remove:
                print(f"Removing the following covariates: {f2remove}")
                self.__covariate_matrix.drop(columns=f2remove, inplace=True)
        return self 
    
    def select_covariates(self, covariates):
        covs2use = {c.lower() for c in covariates}
        return self.remove_covariates(set(self.cov_matrix.columns) - covs2use)

    def fix_missing_values(self):
        if self.cov_matrix.isnull().sum().sum() > 0:
            self.__covariate_matrix = self.cov_matrix.fillna(self.cov_matrix.mean())

    def encode_categorical(self, covariate_info):
        #read covariate info file 
        cov_info = pd.read_csv(covariate_info, sep="\t", index_col=0, names=["Type"])
        #extract categorical feature names
        categoricals = {cov.lower() for cov in cov_info[cov_info["Type"] == "factor"].index}
        #keep only covariates that are present in the matrix    
        categoricals = list(categoricals.intersection(self.cov_matrix.columns))
        categories = list() 

        binary_covariates = list() 
        more_than_binary_covariates = list()
        dataframe_columns = list() 

        for cov in categoricals:
            unique_values = self.cov_matrix[cov].unique()

            if len(unique_values) > 2:
                categories.append(unique_values)
                more_than_binary_covariates.append(cov)
                for cat in unique_values:
                    dataframe_columns.append("{}_{}".format(cov, cat))
            else:
                binary_covariates.append(cov)

        ohe = OneHotEncoder(handle_unknown="ignore", categories=categories).fit(
            self.cov_matrix[more_than_binary_covariates])
        ohe_binary = OneHotEncoder(handle_unknown="error", drop="first").fit(
            self.cov_matrix[binary_covariates]
        )

        self.__covariate_matrix = pd.concat([
            #continuous features 
            self.__covariate_matrix.drop(columns=categoricals),
            #other categorical covariates 
            pd.DataFrame(
                index = self.cov_matrix.index, 
                columns = dataframe_columns, 
                data = ohe.transform(self.cov_matrix[more_than_binary_covariates]).toarray()
            ), 
            #binary covariates
            pd.DataFrame(
                index = self.cov_matrix.index, 
                columns = binary_covariates, 
                data = ohe_binary.transform(self.cov_matrix[binary_covariates]).toarray()
            )], 
            axis=1
        )

    def select_features(self, feature_file = None):
        """ """
        dataset = Dataset()
        dataset.__target = self.__target 
        dataset.__target_labels = self.__target_labels

        covariates = pd.read_csv(feature_file, index_col=0, sep="\t").index
        dataset.__covariate_matrix = self.__covariate_matrix[covariates]
        print(f"Covariate matrix filtered. The following covariates have been selected from {feature_file}")
        
        
        return dataset