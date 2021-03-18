import numpy as np 
import enum
import abc 

from sklearn.pipeline import Pipeline
from sklearn.linear_model import \
    LogisticRegression, \
    SGDClassifier, \
    LassoCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import \
    RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.feature_selection import \
    VarianceThreshold, \
    SelectKBest, \
    SelectFromModel, \
    f_classif, \
    mutual_info_classif    
from sklearn.preprocessing import \
    StandardScaler, \
    RobustScaler, \
    MinMaxScaler


class ClassifiersToEvaluate(enum.Enum):
    """ List of the classifiers to hypertune """
    LOGISTIC_REGRESSION = ("log_reg", LogisticRegression)
    RANDOM_FOREST = ("r_forest", RandomForestClassifier)
    GRADIENT_BOOSTING = ("g_boost", GradientBoostingClassifier)
#    LINEAR_SVM = ("lin_SVM", LinearSVC) ### do not converge
#    SDG = ("sdg", SGDClassifier)       ### do not converge 


class FeatureSelectionHyperParameters:
    @classmethod
    def get_params(cls, pipeline, max_features):
        fs_params = {
            VarianceThreshold: cls.varianceThresholdParameters, 
            StandardScaler: cls.scalerParameters, 
            SelectKBest: cls.selectKBestParameters, 
            SelectFromModel: cls.selectFromModelParameters
        }

        dict_params = dict()

        for step in pipeline[:-1]:
            step_t = type(step)
            step_params_method = fs_params[step_t]

            try:
                dict_params.update(step_params_method())
            except TypeError:
                dict_params.update(step_params_method(max_features))

        return dict_params

    @classmethod
    def varianceThresholdParameters(cls):
        return dict(
            var_threshold__threshold = [0, 0.5, 1]
        )

    @classmethod
    def scalerParameters(cls):
        return dict(
            scaler = [StandardScaler(), RobustScaler(), MinMaxScaler(), "passthrough"]
        )

    @classmethod
    def selectKBestParameters(cls, max_features):
        """ Return a dictionary containing the SelectKBest's hyperparameters """
        return dict(
            selector__score_func = [f_classif, mutual_info_classif], 
            selector__k = list(range(5, max_features, 5)) + ["all"]
        )
    
    @classmethod
    def selectFromModelParameters(cls, max_features):
        """ Return a dictionary containing the SelectFromModel's hyperparameters """
        return dict(
            selector__estimator = [
                SGDClassifier(loss="log", max_iter=10000), 
                SGDClassifier(max_iter=10000), 
                GradientBoostingClassifier()
            ], 
            selector__max_features = [max_features], 
            selector__threshold = ["mean", "median", None]
        )

class ClassifiersHyperParameters:
    @classmethod
    def get_params(cls, pipeline, max_features):
        """ Return a dictionary containing the hyperparameters 
        of the whole pipeline """ 

        clf_params = {
            LogisticRegression: cls.logisticRegressionParameters, 
            LinearSVC: cls.svmParameters, 
            RandomForestClassifier: cls.randomForestParameters, 
            GradientBoostingClassifier: cls.gradientBoostingParameters, 
            SGDClassifier: cls.sdgClassifierParameters
        }
        # get estimator's hyperparameters
        estimator_t = type(pipeline[-1])
        clf_hp = clf_params[estimator_t]()
        # get the hyperparameters of the rest of the pipeline 
        fs_hp = FeatureSelectionHyperParameters.get_params(pipeline, max_features)

        return {**fs_hp, **clf_hp}



    @classmethod
    def logisticRegressionParameters(cls):
        
        return dict(
            log_reg = [LogisticRegression()], 
            log_reg__C = [0.1, 1, 10, 100, 1000], 
            log_reg__penalty = ["l1", "l2"], 
            log_reg__solver = ["liblinear"], 
            log_reg__max_iter = [10000], 
            log_reg__dual = [False]
        )

    @classmethod
    def svmParameters(cls):
        return dict(
            lin_SVM = [LinearSVC()], 
            lin_SVM__penalty = ["l1", "l2"], 
            lin_SVM__loss = ["hinge", "squared_hinge"], 
            lin_SVM__C = [0.1, 1, 10, 100, 1000], 
            lin_SVM__dual = [False], 
            lin_SVM__max_iter = [10000]
        )
    
    @classmethod
    def randomForestParameters(cls):
        return dict(
            r_forest = [RandomForestClassifier()], 
            r_forest__n_estimators = [25, 50, 100, 200, 500, 1000], 
            r_forest__criterion = ["gini", "entropy"],
            r_forest__max_depth = np.arange(3, 10), 
            r_forest__min_samples_split = np.arange(2, 6)
        )

    @classmethod
    def gradientBoostingParameters(cls):
        return dict(
            g_boost = [GradientBoostingClassifier()], 
            g_boost__loss = ["deviance", "exponential"], 
            g_boost__n_estimators = [100, 150, 200, 300], 
            g_boost__subsample = [0.1, 0.5, 1], 
            g_boost__max_depth = [3,4,5]
        )

    @classmethod
    def sdgClassifierParameters(cls):
        return dict(
            sdg = [SGDClassifier()], 
            sdg__loss = ["hinge", "squared_hinge", "log", "modified_huber"], 
            sdg__penalty = ["l1", "l2"], 
            sdg__max_iter = [10000]        
        )

 

class AbstractPipeline(abc.ABC):
    def __init__(self, dataset, pipeline_steps):
        self.__n_features = dataset.shape[1]
        self.__pipelines = [Pipeline([
            #unpack previous pipeline steps
            *pipeline_steps, 
            #add estimator 
            (clf_entry.value[0], clf_entry.value[1]())]) \
                for clf_entry in ClassifiersToEvaluate]

    def get_pipelines(self):
        """ Returns a list of pairs (pipeline, pipeline_params) 
        in order to perform hypertuning """

        return [
            (pipeline, ClassifiersHyperParameters.get_params(pipeline, self.__n_features)) \
                for pipeline in self.__pipelines
        ]


class KBestEstimator(AbstractPipeline):
    def __init__(self, dataset, k=None):
        if k is None:
            k = dataset.shape[1] // 2
            while k > 100:
                k //= 2

        super().__init__(dataset, [
            ("var_threshold", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(k=k))]
        )



class FromModelEstimator(AbstractPipeline):
    def __init__(self, dataset, k = None):
        estimator = LogisticRegression(max_iter=10000)
        if k is None:
            super().__init__(dataset, [
                ("var_threshold", VarianceThreshold()),
                ("scaler", StandardScaler()), 
                ("selector", SelectFromModel(estimator))
            ])
        else:
            super().__init__(dataset, [
                ("var_threshold", VarianceThreshold()),
                ("scaler", StandardScaler()), 
                ("selector", SelectFromModel(estimator, max_features=k, threshold=-np.inf))
            ])


class EstimatorWithoutFS(AbstractPipeline):
    def __init__(self, dataset):
        super().__init__(dataset, [
            ("var_threshold", VarianceThreshold()), 
            ("scaler", StandardScaler())
        ])