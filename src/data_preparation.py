#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
#import seaborn as sns 
import matplotlib.pyplot as plt  

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter  

class MultipleLabelsException(Exception):
    def __init__(self, labels):
        message = (
            f"The dataset contains more than 2 labels ({labels}).\n"
            "Please specify the --binary option.")
        super().__init__(message)
 
class UnsupportedFileType(Exception):
    def __init__(self, filename):
        message = f"Unsupported file type => {filename}"
        super().__init__(message)


#load dataframes from files also allowing DataFrame objects
def fix_input(list_of_stuff: list, transpose=False):
    result = list() 
    for stuff in list_of_stuff:
        if isinstance(stuff, pd.DataFrame):
            result.append(stuff)
        elif isinstance(stuff, str):
            df = pd.read_csv(stuff, sep="\t", index_col=0)
            result.append(df if transpose is False else df.T)
    
    return [prepare_feature_names(x) for x in result] 

def prepare_feature_names(df: pd.DataFrame):
    if isinstance(df, pd.DataFrame):
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace(":", "_")
    elif isinstance(df, pd.Series):
        df = df.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace(':', '_')
    return df 


class Dataset:
    def __init__(self, count_matrices = None, covariate_matrices = None, X = None, y = None):
        self.__covariate_matrix = None ## XXX to rename
        self.__target = None 
        self.__target_labels = None 
        self.__features_names = None 

        if isinstance(count_matrices, list) and isinstance(covariate_matrices, list):
            covariates = pd.concat(fix_input(covariate_matrices))
            counts = pd.concat(fix_input(count_matrices, transpose=True))

            if len(counts) > 0:
                covariates = pd.merge(covariates, counts, left_index=True, right_index=True)
            
            self.__covariate_matrix = covariates

        elif isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray): #X is not None and y is not None:
            self.__covariate_matrix = X 
            self.__target = y 

    
    def get_train_validation(self, size_val=0.1, num_rep=1):
        #XXX - build as generator 
        all_data = self.data.copy()
        target_col = "classification_target"
        all_data[target_col] = self.target 

        for n in range(num_rep):
            validation = pd.concat([
                df.sample(frac=size_val) for _, df in all_data.groupby(target_col)
            ])
            training = pd.concat([all_data, validation]).drop_duplicates(keep=False)

            t = Dataset(X=training.drop(columns=[target_col]), y = training[target_col].to_numpy())
            v = Dataset(X=validation.drop(columns=[target_col]), y = validation[target_col].to_numpy())

            yield t, v


    @classmethod
    def load_input_data(cls, input_data, cov2predict, target_labels, covs2use = None, covs2ignore = None, index_column = None):
        dataframes = list()

        #load data 
        if input_data.endswith(".xlsx"):
            #load data from excel file to one or more dataframes 
            with pd.ExcelFile(input_data) as xlsx:
                for sheet_name in xlsx.sheet_names:
                    dataframes.append((sheet_name, pd.read_excel(xlsx, sheet_name=sheet_name)))

        elif any([input_data.endswith(ext) for ext in (".csv", ".tsv", ".txt")]):
            separator = "," if input_data.endswith(".csv") else "\t"
            dataframes.append((input_data, pd.read_csv(input_data, sep=separator, index_col=0)))

        else:
            raise UnsupportedFileType(input_data)

        #preprocess data:
        #remove useless cols, extract target to predict, encode categorical etc 
        datasets = dict() 
        for name, df in dataframes:
            #normalise column names 
            df.columns = df.columns.str.lower() 

            if index_column:
                df.set_index(index_column.lower(), inplace=True)
            
            bad_cols = None 
            if covs2use is None:
                #remove unnecessary columns
                bad_cols = [col for col in df.columns if col.startswith("unnamed:")]
                if covs2ignore:
                    bad_cols.extend([col.lower() for col in covs2ignore])
            else:
                #remove all features except those in covs2use list 
                whole_set = set(df.columns)
                whole_set.remove(cov2predict)
                bad_cols = whole_set.difference({x.lower() for x in covs2use})

            if bad_cols:
                df.drop(columns=bad_cols, inplace=True)


            #fix numerical values and encode categorical features
            cov2predict = cov2predict.lower()
            for col in df.columns:
                if col != cov2predict:
                    try:
                        df[col] = df[col].apply(lambda x: float(str(x).split()[0].replace(",", "")))
                        df[col].astype("float64").dtypes 
                    except ValueError:
                        #probably we encountered a categorical feature 
                        df[col] = df[col].astype("category")
                        df[col] = df[col].cat.codes
            #fill missing values     
            if df.isnull().sum().sum() > 0:
                df.fillna(df.mean(), inplace=True)
        
            #remove strange characters from features names
            dataset = Dataset()
            dataset.__covariate_matrix = prepare_feature_names(df)

            dataset.filter_by_target(cov2predict, target_labels)

            datasets[name] = dataset 


        return datasets 



    @classmethod
    def read_from_excel(cls, excel_filename, cov2predict, target_labels, covs2use = None, covs2ignore = None, index_column = None):
        datasets = dict() 

        with pd.ExcelFile(excel_filename) as xlsx:
            cov2predict = cov2predict.lower()

            for sheet_name in xlsx.sheet_names: 
                dataset = Dataset() 
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                df.columns = df.columns.str.lower()

                if index_column:
                    df.set_index(index_column.lower(), inplace=True)

                bad_cols = [col for col in df.columns if col.startswith("unnamed:")]
                if covs2ignore:
                    bad_cols.extend([col.lower() for col in covs2ignore])
                if bad_cols:
                    df.drop(columns=bad_cols, inplace=True)

                for colname in df.columns:
                    if colname != cov2predict:
                        try:
                            df[colname] = df[colname].apply(lambda x: float(str(x).split()[0].replace(",", "")))
                            df[colname].astype("float64").dtypes
                        except ValueError:
                            #probably we encountered a categorical feature 
                            df[colname] = df[colname].astype("category")
                            df[colname] = df[colname].cat.codes 
                            

                if df.isnull().sum().sum() > 0:
                    df.fillna(df.mean(), inplace=True)
                
            # print("Plotting {} correlation matrix ".format(sheet))

            # f, ax = plt.subplots(figsize=(10, 8))
            # corr = dataset.cov_matrix.corr().abs()
        
            # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
            # plt.savefig("{}_correlation_matrix.png".format(sheet), dpi = 400)

                df.columns = df.columns.str.replace(' ', '_')
                df.columns = df.columns.str.replace('-', '_')
                df.columns = df.columns.str.replace(":", "_")
    
                dataset.__covariate_matrix = df
                

                # ####
                # dfff = dataset.__covariate_matrix
                # h = dfff.loc[dfff["id_pathology"] == "Healthy"]
                # crc = dfff.loc[dfff["id_pathology"] == "CRC"]

                # print(h.describe())

                # print(crc.describe())

                # with pd.ExcelWriter("pippo.xlsx", mode="w") as writer:
                #     crc.describe().to_excel(
                #         writer, sheet_name="crc", float_format="%.3f", index=True)

                #     h.describe().to_excel(
                #         writer, sheet_name="h", float_format="%.3f", index=True)

                # # raise Exception("fine")  

                ####

                dataset.filter_by_target(cov2predict, target_labels)

                datasets[sheet_name] = dataset 

        return datasets 
    
#     @classmethod
#     def do_preprocessing(cls, count_matrices, covariate_matrices, covs_types, cov2predict, target_labels, covs2use, covs2ignore):
#         dataset = Dataset(count_matrices, covariate_matrices)

#         if covs2use:
#             f_to_keep = covs2use + [cov2predict]
#             dataset.select_covariates(f_to_keep)
#         elif covs2ignore:
#             dataset.remove_covariates(covs2ignore)
        
#         dataset.filter_by_target(cov2predict, target_labels)
#         dataset.fix_missing_values() #not implemented 
# #        dataset.encode_categorical(covs_types)

#         return dataset 
    
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

        print(f"Target: {target_cov}\nLabels: {allowed_values}")
        # print(self.data.columns)
        # print(self.data)

        if not allowed_values:
            allowed_values = set(self.cov_matrix[target_cov])
            if len(allowed_values) != 2:
                raise MultipleLabelsException(allowed_values)

        self.__target_labels = allowed_values
        mask = self.cov_matrix[target_cov].isin(allowed_values)
        cov_masked = self.cov_matrix[mask]

        self.__target_encoding = {label: encoding for encoding, label in enumerate(allowed_values)}
        self.__target = cov_masked[target_cov].replace(self.__target_encoding).to_numpy()
        self.__covariate_matrix = cov_masked.drop(columns=[target_cov])
    

        print(f"\nTarget labels encoded with the following mapping: {self.__target_encoding}")

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
                    dataframe_columns.append(f"{cov}_{cat}")
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

    def select_features(self, feature_file=None, feature_list=None):
        """ """
        flist = feature_list if feature_list else \
                        pd.read_csv(feature_file, index_col=0, sep="\t") \
                            if feature_file else None
        if flist is None:
            raise Exception("No feature list nor file provided. ")
            
        dataset = Dataset()
        dataset.__target = self.__target 
        dataset.__target_labels = self.__target_labels


        covariates = prepare_feature_names(flist.index.to_series())

        dataset.__covariate_matrix = self.__covariate_matrix[covariates]
        print(f"Covariate matrix filtered (file={feature_file})")
        
        
        return dataset