import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import math
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy import stats
import itertools as it
import sklearn.metrics as metrics
import sklearn.model_selection as selection
import os
import json
import pickle as pic
import importlib
import sys
sys.path.append("./../utils")
sys.path.append("./../sklearn")
import util
import constants as const
import ColumnTransformer
importlib.reload(util)
importlib.reload(const)
importlib.reload(ColumnTransformer)
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('max_rows', None)
pd.set_option('precision', 6)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import json
import pickle
from sklearn.metrics import make_scorer
import datetime
import time


TOTAL_TIME = "total_time"
PROCESSING_TIME = "processing_time"
REP_FACTOR = "replication_factor"

def my_scorer(y_true, y_predicted):
    """mape score function

    Args:
        y_true (array-like): true values
        y_predicted (array-like): predicted values

    Returns:
        float: the mape score
    """
    epsilon = np.finfo(np.float64).eps
    output_errors = np.average((
        np.abs(y_predicted - y_true) / np.maximum(np.abs(y_true), epsilon)
    ), axis=0)
    return np.average(output_errors)


my_func = make_scorer(my_scorer, greater_is_better=False)

class Metrics():
    """ Metrics class which can be used to calculcated different scores. 
    """

    def __init__(self, target: str, predicted_target: str):
        """ constructor

        Args:
            target (str): name of column which contains the true values of the target
            predicted_target (str): name of column which contains the predicted values of the target
        """
        self.target = target
        self.predicted_target = predicted_target

    def get_metrics(self, data) -> dict:
        """ Calculated different metrics. 

        Args:
            data (dataframe): the date the contains the true and the predicted target values

        Returns:
            dict: the scores 
        """
        y_true = data[self.target]
        y_predicted = data[self.predicted_target]

        r2 = metrics.r2_score(y_true, y_predicted)

        rmse = metrics.mean_squared_error(y_true, y_predicted, squared=False)

        epsilon = np.finfo(np.float64).eps
        output_errors = np.average((
            np.abs(y_predicted - y_true) / np.maximum(np.abs(y_true), epsilon)
        ), axis=0)
        mape = np.average(output_errors)
        return {
            'r2': r2,
            'rmse': rmse,
            'mape': mape,
        }

scaler = 1
def target_transform(target):
    target_ = target.copy()
    target_ = target_ * scaler
    return target_

def inverse_target_transform(target):
    target_ = target.copy()
    target_ = target_ / scaler
    return target_

def get_general_pipline(
    feature_combinations, 
    regressor_steps, 
    grid_search):
    """ Create a pipeline. For all pipelines in a first step the features are selected from the feature_combinations and in a second step the data is scaled.

    Args:
        feature_combinations (list((list))): feature_combinations. Here you can set different combindation
        regressor_steps (list(tuples)): Add further steps. Most important add a step for the regressor to apply
        grid_search (dict): the parameter search space
    """

    steps = [
        ('feature_selection', ColumnTransformer.ColumnTransformer(feature_combinations)),
        ('scaler', StandardScaler())
    ]

    # Add model specific steps
    steps += regressor_steps

    # Create the Pipeline 
    pipeline = Pipeline(steps=steps)

    pipeline_transformed = TransformedTargetRegressor(
        regressor=pipeline, 
        func=target_transform, 
        inverse_func=inverse_target_transform
    )

    m =  GridSearchCV(
        pipeline_transformed, 
        grid_search, 
        scoring=my_func,
        n_jobs=-1, 
        verbose=10, 
        cv=5
    )
   # print("PARAMETERS", m.get_params().keys())
    return m

def get_rfr_pipeline(feature_combinations, grid_search):
    """Create the random forests pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """
    regressor_steps = [("regressor", RandomForestRegressor())]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)


def get_svr_pipeline(feature_combinations, grid_search ):
    """Create the support vector regression pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """

    #regressor_steps = ("regressor", SVR(kernel='linear',  C=1, epsilon=0.01))
    regressor_steps = [("regressor", SVR())]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)

def get_xgb_pipeline(feature_combinations, grid_search):
    """Create the extreme gradient bossting (xgb) pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """
    regressor_steps = [('regressor',  XGBRegressor(n_estimators=800, max_depth=15, objective='reg:squarederror'))]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)

def get_polynomial_regression(feature_combinations, grid_search):
    """Create the polynomial regression pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """
    regressor_steps = [
        ('poli', PolynomialFeatures()),
        ('regressor', LinearRegression())
    ]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)

def get_linear_regression(feature_combinations, grid_search):
    """Create the linear regression pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """

    regressor_steps = [
        ('regressor', LinearRegression())
    ]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)

def get_knn(feature_combinations, grid_search):
    """Create the knnn pipeline

    Args:
        feature_combinations (list): the features combinations
        grid_search (dict): the parameter space

    Returns:
        pipeline: the pipeline
    """

    regressor_steps = [
        ('regressor', KNeighborsRegressor())
    ]
    return get_general_pipline(feature_combinations=feature_combinations, regressor_steps=regressor_steps, grid_search=grid_search)


def get_model(model_name, feature_combinations):
    """ Get the whole pipeline for the given model_name

    Args:
        model_name (str): the model that should be used
        feature_combinations (list): the features combinations 

    Returns:
        pipeline: The pipline with the grid search
    """
    if (model_name == "SVR"):
        grid_search = {}
        grid_search['regressor__feature_selection__features'] = feature_combinations
        grid_search['regressor__regressor__kernel'] =  ["linear"]
        grid_search['regressor__regressor__C'] =  np.linspace(0.001, 1, 5)
        grid_search['regressor__regressor__epsilon'] =  np.linspace(0.001, 0.1, 5)
        
      #  grid_search = {}
      #  grid_search['regressor__feature_selection__features'] = feature_combinations
      #  grid_search['regressor__regressor__kernel'] =  ["linear"]
      #  grid_search['regressor__regressor__C'] =  [1]
      #  grid_search['regressor__regressor__epsilon'] =  [0.01]
        return get_svr_pipeline(feature_combinations=feature_combinations, grid_search=grid_search) 
    if (model_name == "RFR"):
      #  list(range(25,525,25)),
       # list(range(2,40,2))
        grid_search = {
       # 'regressor__feature_selection__features': [250],
        'regressor__regressor__n_estimators': [100, 300, 500],
        'regressor__regressor__max_depth': [10, 20 , 30]
        }
        grid_search['regressor__feature_selection__features'] = feature_combinations
        return get_rfr_pipeline(feature_combinations=feature_combinations, grid_search=grid_search)
    if (model_name == "XGB"):
        grid_search = {
            "regressor__regressor__n_estimators": [100, 300, 500],
            "regressor__regressor__max_depth":[10, 20, 30],
         #   "regressor__regressor__objective":['reg:squarederror'],
        }
        grid_search['regressor__feature_selection__features'] = feature_combinations
        return get_xgb_pipeline(feature_combinations=feature_combinations, grid_search=grid_search)
    if (model_name == "PolyRegression"):
        grid_search = {
            'regressor__feature_selection__features': feature_combinations,
            "regressor__poli__degree":[1,2],
        }
        return  get_polynomial_regression(feature_combinations=feature_combinations, grid_search=grid_search)
    if (model_name == "LinearRegression"):
        grid_search = {
        }
        grid_search['regressor__feature_selection__features'] = feature_combinations
        return  get_linear_regression(feature_combinations=feature_combinations, grid_search=grid_search)
    if model_name == "KNN":
        grid_search = {
        #'regressor__feature_selection__features': features,
       'regressor__regressor__n_neighbors':list(range(1,5,1)),
        'regressor__regressor__p': list(range(1,5,1)),
        }
        grid_search['regressor__feature_selection__features'] = feature_combinations
        return  get_knn(feature_combinations=feature_combinations, grid_search=grid_search)
   
def powerset(s):
    """ Creates the powerset of s

    Args:
        s (list): list with feature names

    Returns:
        list: the power set of s
    """
    r = []
    x = len(s)
    for i in range(1 << x):
        r.append([s[j] for j in range(x) if (i & (1 << j))])
    return r
    
def create_Feature_combinations(candidates, canditates_inlcuded):
    """Generate all combinations

    Args:
        candidates (list):  all featurs which are canditates
        canditates_inlcuded (list): all featurs which will always be included
    Returns:
            list: all feature combinations
    """
    final_canditates = []
    for c in candidates:
        if not c in canditates_inlcuded:
            final_canditates.append(c)
    all_combinations = powerset(final_canditates)

    for combination in all_combinations:
        combination += canditates_inlcuded
    return all_combinations


def get_one_hot_encoded_data(data):
    """One hot encoding of partitioner.
    
    Parameters
    ----------
    data: a dataframe containing the column partitioner which will be one hot encoded.
    
    Returns
    -------
    data : the one hot encoded dataframe.
    """  
    df_dummies = pd.get_dummies(data['partitioner'])
    df_new = pd.concat([data, df_dummies], axis=1)
    return df_new

def get_time_stamp():
    """ Get the current day and time as a string

    Returns:
        str: day and time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')

def create_results_folder(base_directory):
    """ Create a directory in the base directory with the day ans time as the directory name

    Args:
        base_directory (str): in which directory to create the directory

    Returns:
        str: the path to the created directory
    """
    dir_name = "{}/{}".format(base_directory, get_time_stamp())
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
    except OSError:
        print ('Error: Creating directory. ' +  dir_name)
        

import glob
import pickle as pic
import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (3,1)
from sklearn import pipeline
import constants
import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import itertools as it
import sklearn.metrics as metrics
import sklearn.model_selection as selection
import os
import config
import json
import pickle
import importlib
from xgboost import XGBRegressor
import time
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
def load_partitioning_time(timestamp,  best_model_name):
    """Loads the model to predict the partitioning run-time.

    Args:
        timestamp (str): The timestamp, needed to identify the serialized model
        best_model_name (str): the model name like for example XGB, needed to identify the serialized model

    Returns:
        model: the model to predict partitioning run-times. 
    """
    BASE_PATH = "/home/ubuntu/cephstorage/partitioning-run-time-prediction-models/"
    paths_to_models = glob.glob("{}/{}/*".format(BASE_PATH, timestamp))
    
    # Filter the models 
    filtered = list(filter(lambda paths_to_model: "single-model-for-all-partitioner" in paths_to_model and best_model_name in paths_to_model , paths_to_models))
    
    # Verify the we were able to select the model.
    if len(filtered) != 1:
        print(filtered)
        print("We have a problem. We have NOT selectly one model.")
    
    path_to_model = filtered[0]
    model = pic.load(open(path_to_model, 'rb'))

    # Not needed, but may be needed in the feature. 
    #features = model.best_params_["regressor__feature_selection__features"]
    return model

def load_processing_time(timestamp,  best_model_name, algorithm):
    """Loads the model to predict the processing run-time for the given algorithm.

    Args:
        timestamp (str): The timestamp, needed to identify the serialized model
        best_model_name (str): the model name like for example XGB, needed to identify the serialized model
        algorithm (str): The graph processing algorithm for which to predict the processing time. E.g., pr, cc, ...

    Returns:
        model: the model to predict processing run-time of the given graph processing algorithm. 
    """
    algorithm_pattern = "_{}_".format(algorithm)
    BASE_PATH = "/home/ubuntu/cephstorage/graph-processing-run-time-prediction-models"
    paths_to_models = glob.glob("{}/{}/*".format(BASE_PATH, timestamp))
    filtered = list(filter(lambda paths_to_model: ("single-model-for-all-partitioner" in paths_to_model) and (best_model_name in paths_to_model) and algorithm_pattern in paths_to_model , paths_to_models))
    if len(filtered) != 1:
        print(filtered)
        print("We have a problem. We have NOT selectly one model. ")
    path_to_model = filtered[0]
    model = pic.load(open(path_to_model, 'rb'))
    # Not needed, but may be needed in the feature. 
    #features = model.best_params_["regressor__feature_selection__features"]
    return model


class GraphLearner():
    """ This class represents GraphLearner. It can be used to predict:
        1. Graph Partitioning Run-times
        2. Graph Processing Run-time
        3. Graph Partitioning Quality

        Furthermore, the class can be used to evaluate our two use cases for GraphLearner:
        1. Select the partitioner which leads to the fastest end-to-end time
        2. Select the partitioner which leads to the fastest processing time. 
    """

    def __init__(
        self, 
        partitioning_model_name, 
        partitioning_model_timestamp, 
        processing_model_description, 
        processing_model_timestamp, 
        quality_models_descriptions,
        number_iterations):
        """ Constructor for GraphLearner.

        Args:
            partitioning_model_name (str): The model for the partitioning time prediction. E.g., XGP
            partitioning_model_timestamp (str): The timestamp, needed to identify the serialized model
            processing_model_description list(str, str): For which graph processing should we select which model. e.g, [("pr", "XGB"), ("cc", "SVR")]
            processing_model_timestamp (str): The timestamp, needed to identify the serialized model
            quality_model_description (str): We trained the graph quality predict with two features sets, "Easy" and "Hard". You select between those two
            number_iterations (int): We have algorithms (cc, sssp, kcore) for which we predict the overall run-time and algorithms (pr and the two synthetic algorithms) for which we predict the average iterationt time.
                For the algorithms of the latter category we need to specify for how many iterations the algorithm should be executed. 
        """
        self.partitioning_runtime_model = load_partitioning_time(timestamp=partitioning_model_timestamp, best_model_name=partitioning_model_name)
        self.processing_runtime_models = {}
        self.quality_models = {}

        def get_quality_model_path(timestamp, model_name, featureset, enrichment_level, target):
            #return "/home/ubuntu/cephstorage/partitioning-quality-prediction/{}_{}-{}-{}-{}".format(timestamp, model_name, featureset, enrichment_level, target)
            return "/home/ubuntu/cephstorage/projects/graphlearner/quality/serialized_models/{}_{}-{}-{}-{}".format(timestamp, model_name, featureset, enrichment_level, target)
            

        for quality_models_description in quality_models_descriptions:
            path_to_model = get_quality_model_path(
                quality_models_description["timestamp"], 
                quality_models_description["model_name"], 
                quality_models_description["featureset"], 
                quality_models_description["enrichment_level"], 
                quality_models_description["target"], 
            )

            self.quality_models[quality_models_description["target"]] = pic.load(open(path_to_model, "rb"))

        self.num_iterations = number_iterations
        # We train one model per graph processing algorithm
        for description in processing_model_description:
            algorithm = description["algorithm"]
            model_name = description["model_name"]
            self.processing_runtime_models[algorithm] = load_processing_time(timestamp=processing_model_timestamp, best_model_name=model_name, algorithm=algorithm)
    

    def predict_quality(self, data):
        _data = data.copy() 
        _data["replication_factor"] = self.quality_models["replication_factor"].predict(_data)
        _data["edge_balance"] = self.quality_models["edge_balance"].predict(_data)
        _data["source_balance"] = self.quality_models["source_balance"].predict(_data)
        _data["destination_balance"] = self.quality_models["destination_balance"].predict(_data)
        _data["vertex_balance"] = self.quality_models["vertex_balance"].predict(_data)
        return _data


    def get_processing_runtime_model(self, algorithm):
        return self.processing_runtime_models[algorithm]
    
    def get_partitioning_runtime_model(self):
        return self.partitioning_runtime_model

    def get_quality_model(self, target):
        return self.quality_models[target]

    def set_number_iterations(self, number_iterations):
        self.num_iterations = number_iterations

    def get_num_iterations(self, algorithm):
        """ As described above their are two categories of algorithms. One were we predict the overall graph processing time and one were we only predict the mean iteration time. 

        Args:
            algorithm (str): The graph processing algorithm

        Returns:
            int: the number of iterations
        """
        if algorithm in ["cc", "kcoreavg", "sssp1"]:
            return 1 # because we predict the sum sum * 1 is still the sum. 
        else:
            return self.num_iterations #because we predict the average iteration time. So we need to multiply the prediction.

    def select(self, data, algorithm):
        _data = data.copy()
        _data = _data[_data.algorithm == algorithm]
        _data = self.predict_quality(_data)    
        _data = self.predict_times(data=_data, algorithm=algorithm)     
        e2e =  _data.sort_values(["predicted_total_time"])["partitioner"].to_numpy()[0]
        process =  _data.sort_values(["predicted_processing_time"])["partitioner"].to_numpy()[0]
        return {"process":process, "e2e": e2e}


    def predict(self, data, algorithm):
        _data = data.copy()
        _data = _data[_data.algorithm == algorithm]
        _data = self.predict_quality(_data)    
        _data = self.predict_times(data=_data, algorithm=algorithm)        
        _data["processing_time"] =  self.get_num_iterations(algorithm=algorithm) * _data["processing_time"] # alrady in seconds 
        _data["processing_time"] = _data["processing_time"] / 1000 # ms to second
        _data["total_time"] = _data["partitioning_time"] +  _data["processing_time"]
        return _data

    
    def predict_partitioning_runtime(self, data):
        return self.partitioning_runtime_model.predict(data) # already in seconds

    def predict_processing_runtime(self, data, algorithm):
        return (self.get_num_iterations(algorithm=algorithm) *  self.processing_runtime_models[algorithm].predict(data)) / 1000  # ms to seconds

    def predict_times(self, data, algorithm):
        _data = data.copy()
        _data["predicted_partitioning_time"] = self.partitioning_runtime_model.predict(data) # already in seconds
        _data["predicted_processing_time"] = self.get_num_iterations(algorithm=algorithm) *  self.processing_runtime_models[algorithm].predict(data)
        _data["predicted_processing_time"] =  _data["predicted_processing_time"]  / 1000  # ms to seconds
        _data["predicted_total_time"] = _data["predicted_partitioning_time"] + _data["predicted_processing_time"]
        return _data

    def evaluate(self, data, partitioners):
        """Evaluate the data that contains the true values and the predicted ones. 

        Args:
            data (datafrane): the data that contains the true values and the predicted ones.
            partitioners (list): The partitions to consider

        Returns:
            list: results
        """
        results = []
        for algorithm in data.algorithm.unique():
            for graph in data.graph.unique():
                _data = data.copy()
                _data = _data[(_data.algorithm == algorithm) & (_data.graph == graph) & (_data.partitioner.isin(partitioners))]
                # We sort based on the ground truth values
                sorted_values_truth_df = _data.sort_values(by=[TOTAL_TIME])
                sorted_values_truth = sorted_values_truth_df[TOTAL_TIME].to_numpy()
                best_time = sorted_values_truth[0]
                mean_time = np.mean(sorted_values_truth)
                worst_time = sorted_values_truth[-1]
                best_partitioner = sorted_values_truth_df["partitioner"].to_numpy()[0]
                sorted_values_lowest_replication_factor_df = _data.sort_values(by=[REP_FACTOR])
                sorted_values_lowest_replication_factor = sorted_values_lowest_replication_factor_df[TOTAL_TIME].to_numpy()
                lowest_rep_time = sorted_values_lowest_replication_factor[0]
                sorted_values_df = _data.sort_values(by=["predicted_total_time"]) 
                sorted_values = sorted_values_df[TOTAL_TIME].to_numpy() #  to which time does the selection lead.
                selected_time = sorted_values[0] # At index 0: the predicted fastest time
                values_for_plot_partitioners = list(sorted_values_truth_df["partitioner"].to_numpy())  + ["1A"]
                values_for_plot_times = list(sorted_values_truth_df[TOTAL_TIME].to_numpy()) + [selected_time]
                values_for_plot_partitioners = values_for_plot_partitioners + ["RS"]
                values_for_plot_times = values_for_plot_times + [lowest_rep_time]
                values_for_plot_df = pd.DataFrame({"partitioner": values_for_plot_partitioners, TOTAL_TIME: values_for_plot_times}).sort_values(by=[TOTAL_TIME, "partitioner"])
                p_colors = values_for_plot_df["partitioner"].to_numpy()
                colors = ["blue"] * len(p_colors)
                for i in range(len(p_colors)):
                    if p_colors[i] == "1A":
                        colors[i] = "black"
                results.append({
                    "best_partitioner": best_partitioner, 
                    "best_time":best_time, 
                    "worst_time":worst_time,
                    "mean_time":mean_time,
                    "selected_time":selected_time, 
                    "selected/best":selected_time/best_time,
                    "selected/mean": selected_time/mean_time,
                    "selected/worst":selected_time/worst_time ,
                    "selected/lowest_rep_time":selected_time/lowest_rep_time,
                    "best/mean":best_time/mean_time ,
                    "best/worst": best_time/worst_time,
                    "lowest_rep_time/best":lowest_rep_time/best_time,
                    "lowest_rep_time/mean": lowest_rep_time/mean_time,
                    "lowest_rep_time/worst": lowest_rep_time/worst_time ,
                    "algorithm": algorithm,
                    "times": values_for_plot_df["total_time"].to_numpy(),
                    "colors": colors,
                    "partitioners": values_for_plot_df["partitioner"].to_numpy(),
                    "graph": graph ,
                    "goal": "end2end"
                })

        for algorithm in data.algorithm.unique():
            for graph in data.graph.unique():
                _data = data.copy()
                _data = _data[(_data.algorithm == algorithm) & (_data.graph == graph) & (_data.partitioner.isin(partitioners))]
                # We sort based on the ground truth values
                sorted_values_truth_df = _data.sort_values(by=[PROCESSING_TIME])
                sorted_values_truth = sorted_values_truth_df[PROCESSING_TIME].to_numpy()
                best_time = sorted_values_truth[0]
                mean_time = np.mean(sorted_values_truth)
                worst_time = sorted_values_truth[-1]
                best_partitioner = sorted_values_truth_df["partitioner"].to_numpy()[0]
                sorted_values_lowest_replication_factor_df = _data.sort_values(by=[REP_FACTOR])
                sorted_values_lowest_replication_factor = sorted_values_lowest_replication_factor_df[PROCESSING_TIME].to_numpy()
                lowest_rep_time = sorted_values_lowest_replication_factor[0]
                sorted_values_df = _data.sort_values(by=["predicted_processing_time"])
                sorted_values = sorted_values_df["processing_time"].to_numpy() #  to which time does the selection lead.
                selected_time = sorted_values[0]# the time to which are selection leads
                values_for_plot_partitioners = list(sorted_values_truth_df["partitioner"].to_numpy())  + ["1A"]
                values_for_plot_times = list(sorted_values_truth_df["processing_time"].to_numpy()) + [selected_time]
                values_for_plot_partitioners = values_for_plot_partitioners + ["RS"]
                values_for_plot_times = values_for_plot_times + [lowest_rep_time]
                values_for_plot_df = pd.DataFrame({"partitioner": values_for_plot_partitioners, "processing_time": values_for_plot_times}).sort_values(by=["processing_time", "partitioner"])

                p_colors = values_for_plot_df["partitioner"].to_numpy()
                colors = ["blue"] * len(p_colors)
                for i in range(len(p_colors)):
                    if p_colors[i] == "1A":
                        colors[i] = "black"
                results.append({
                    "best_partitioner": best_partitioner, 
                    "best_time":best_time, 
                    "worst_time":worst_time,
                    "mean_time":mean_time,
                    "selected_time":selected_time, 
                    "selected/best":selected_time/best_time,
                    "selected/mean": selected_time/mean_time,
                    "selected/worst":selected_time/worst_time ,
                    "selected/lowest_rep_time":selected_time/lowest_rep_time,
                    "best/mean":best_time/mean_time ,
                    "best/worst": best_time/worst_time,
                    "lowest_rep_time/best":lowest_rep_time/best_time,
                    "lowest_rep_time/mean": lowest_rep_time/mean_time,
                    "lowest_rep_time/worst": lowest_rep_time/worst_time ,
                    "algorithm": algorithm,
                    "times": values_for_plot_df["processing_time"].to_numpy(),
                    "colors": colors,
                    "partitioners": values_for_plot_df["partitioner"].to_numpy(),
                    "graph": graph ,
                    "goal": "processing"
                })
        return results

FINAL_PARTITIONING_MODEL_NAME = "XGB"
FINAL_PARTITIONING_TIMESTAMP = "2022-03-18-07:49:18"

FINAL_PROCESSING_MODELS = [
         {"algorithm": "cc","model_name": "PolyRegression"}, 
         {"algorithm": "kcoreavg","model_name": "XGB"}, 
         {"algorithm": "pr","model_name": "XGB"},
         {"algorithm": "sssp1","model_name": "XGB"},
         {"algorithm": "synthetic10c0","model_name": "PolyRegression"},
         {"algorithm": "synthetic1c0","model_name": "PolyRegression"}
    ]
FINAL_PROCESSING_TIMESTAMP = "2022-03-18-07:49:44"

FINAL_QUALITY_MODELS_HARD = [
          {"target": "vertex_balance", "timestamp": "2022-03-17-12:53:45", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "destination_balance", "timestamp": "2022-03-17-13:20:23", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "source_balance", "timestamp": "2022-03-17-13:46:39", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "edge_balance", "timestamp": "2022-03-17-14:12:13", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
        {"target": "replication_factor", "timestamp": "2022-03-18-13:50:38", "model_name": "XGB", "featureset": "Hard", "enrichment_level": "0.0"},
     ]

FINAL_QUALITY_MODELS_EASY = [
          {"target": "vertex_balance", "timestamp": "2022-03-17-12:53:45", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "destination_balance", "timestamp": "2022-03-17-13:20:23", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "source_balance", "timestamp": "2022-03-17-13:46:39", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "edge_balance", "timestamp": "2022-03-17-14:12:13", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "replication_factor", "timestamp": "2022-03-18-11:06:38", "model_name": "XGB", "featureset": "Easy", "enrichment_level": "0.0"},
     ]


glearner = GraphLearner(
    partitioning_model_name=FINAL_PARTITIONING_MODEL_NAME, 
    partitioning_model_timestamp=FINAL_PARTITIONING_TIMESTAMP,
    processing_model_description=FINAL_PROCESSING_MODELS,
    processing_model_timestamp=FINAL_PROCESSING_TIMESTAMP,
    quality_models_descriptions=FINAL_QUALITY_MODELS_HARD,
    number_iterations=100)

glearner_no_enrichment = GraphLearner(
    partitioning_model_name="XGB", 
    partitioning_model_timestamp="2022-03-18-07:49:18",
    processing_model_description=[
         {"algorithm": "cc","model_name": "PolyRegression"}, 
         {"algorithm": "kcoreavg","model_name": "XGB"}, 
         {"algorithm": "pr","model_name": "XGB"},
         {"algorithm": "sssp1","model_name": "XGB"},
         {"algorithm": "synthetic10c0","model_name": "PolyRegression"},
         {"algorithm": "synthetic1c0","model_name": "PolyRegression"}
    ],
    processing_model_timestamp="2022-03-18-07:49:44",
    quality_models_descriptions=[
          {"target": "vertex_balance", "timestamp": "2022-03-17-12:53:45", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "destination_balance", "timestamp": "2022-03-17-13:20:23", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "source_balance", "timestamp": "2022-03-17-13:46:39", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "edge_balance", "timestamp": "2022-03-17-14:12:13", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
           {"target": "replication_factor", "timestamp": "2022-03-17-14:59:48", "model_name": "RFR", "featureset": "Hard", "enrichment_level": "0.0"},
     ],
    number_iterations=100)

    
glearner_enrichment = GraphLearner(
    partitioning_model_name="XGB", 
    partitioning_model_timestamp="2022-03-18-07:49:18",
    processing_model_description=[
         {"algorithm": "cc","model_name": "PolyRegression"}, 
         {"algorithm": "kcoreavg","model_name": "XGB"}, 
         {"algorithm": "pr","model_name": "XGB"},
         {"algorithm": "sssp1","model_name": "XGB"},
         {"algorithm": "synthetic10c0","model_name": "PolyRegression"},
         {"algorithm": "synthetic1c0","model_name": "PolyRegression"}
    ],
    processing_model_timestamp="2022-03-18-07:49:44",
    quality_models_descriptions=[
          {"target": "vertex_balance", "timestamp": "2022-03-17-13:18:44", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "1.0"},
          {"target": "destination_balance", "timestamp": "2022-03-17-13:45:01", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "1.0"},
          {"target": "source_balance", "timestamp": "2022-03-17-14:10:44", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "1.0"},
          {"target": "edge_balance", "timestamp": "2022-03-17-14:33:00", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "1.0"},
          {"target": "replication_factor", "timestamp": "2022-03-17-15:29:43", "model_name": "RFR", "featureset": "Hard", "enrichment_level": "1.0"},    
     ],
    number_iterations=100)

glearner_importance = GraphLearner(
    partitioning_model_name="XGB", 
    partitioning_model_timestamp="2022-03-18-07:49:18",
    processing_model_description=[
         {"algorithm": "cc","model_name": "PolyRegression"}, 
         {"algorithm": "kcoreavg","model_name": "XGB"}, 
         {"algorithm": "pr","model_name": "XGB"},
         {"algorithm": "sssp1","model_name": "XGB"},
         {"algorithm": "synthetic10c0","model_name": "PolyRegression"},
         {"algorithm": "synthetic1c0","model_name": "PolyRegression"}
    ],
    processing_model_timestamp="2022-03-18-07:49:44",
    quality_models_descriptions=[
          {"target": "vertex_balance", "timestamp": "2022-03-17-12:53:45", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "destination_balance", "timestamp": "2022-03-17-13:20:23", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "source_balance", "timestamp": "2022-03-17-13:46:39", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "edge_balance", "timestamp": "2022-03-17-14:12:13", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
          {"target": "replication_factor", "timestamp": "2022-03-17-14:34:32", "model_name": "RFR", "featureset": "Easy", "enrichment_level": "0.0"},
     ],
    number_iterations=100)

def get_importance(gl):
    results_dfs = []
    for target in ["replication_factor", "vertex_balance", "destination_balance", "source_balance", "edge_balance"]:
        importance_values = gl.get_quality_model(target).best_estimator_.regressor_.named_steps['regressor'].feature_importances_
        importance_labels = gl.get_quality_model(target).best_params_["regressor__feature_selection__features"]
        r = pd.DataFrame({"feature": importance_labels, "value": importance_values})
        r['feature'] = r['feature'].replace({
            'hep100':'Partitioner',
            'hep10':'Partitioner',
            'hep1':'Partitioner',
            'ne':'Partitioner',
            '2ps':'Partitioner',
            'hdrf':'Partitioner',
            'dbh':'Partitioner',
            '2d':'Partitioner',
            '1dd':'Partitioner',
            '1ds':'Partitioner',
            'crvc':'Partitioner',
            "pearson_mode_degrees_in": "Degree Distr.",
            "pearson_mode_degrees_out": "Degree Distr.",
            "density": "Density",
            "num_partitions": "#Partitions",
            "mean_degree": "Mean Degree"
        })
        r = r.groupby(["feature"], as_index=False).sum().sort_values(by=["value"])
        r = r.rename(columns={"value":target})
                
        results_dfs.append(r)


    accumulate = results_dfs[0]
    for i in range(1,len(results_dfs)):
        accumulate = pd.merge(accumulate, results_dfs[i], on="feature")
    accumulate = accumulate.round(3)
    accumulate
    print(accumulate.sort_values(by="replication_factor", ascending=False).to_latex(index=False))