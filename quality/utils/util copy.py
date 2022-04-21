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




# If set to true, the machine learning models will be serialized for 0% and 100% enrichment. 
STORE_RESULTS = config.STORE_RESULTS
PATH_TO_STORE_MODELS = config.PATH_TO_STORE_MODELS #"/home/ubuntu/cephstorage/graphlearner-models/"


from sklearn.metrics import make_scorer
import datetime
import time


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




def get_time_stamp():
    """ Get the current day and time as a string

    Returns:
        str: day and time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')



def get_path():
    return PATH_TO_STORE_MODELS
    
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

def get_combinations(columns, must_include=None):
    """Create the powerset/all combinations of all elements contained in columns (e.g. features). 
    Thereby, all possible feature combintations are created.
    
    Parameters
    ----------
    columns: the elements for which all combinations should be created.
    must_include: a list of elements which need to be contained in all combinations.
    
    Returns
    -------
    powerset: all combinations.
    """
    if must_include is None:
        must_include = []
    all_combinations = []
    for r in range(1, len(columns) + 1):
        combinations_object = it.combinations(columns, r)
        for tup in combinations_object:
            combinations_list = list(tup)
            all_combinations.append(combinations_list + must_include)
    return all_combinations

scaler = 10000
def target_transform(target):
    target_ = target.copy()
    target_ = target_ * scaler
    return target_

def inverse_target_transform(target):
    target_ = target.copy()
    target_ = target_ / scaler
    return target_

def get_graph_names_of_graph_types(data, graph_types):
    """Get all graph names contained in the given graph types. 
    
    Parameters
    ----------
    data: data of different graphs.
    
    Returns
    -------
    graphs: all graphnames that are contained in the given graph types.  
    """
    condition = data["graph_type"].isin(graph_types)
    selected_names = data[condition].graph.unique()
    return list(selected_names)

def is_real_world_graph(graph_type):
    """Checks if the given graph type is used for real-world graphs. 
    
    Parameters
    ----------
    graphtype: string
        The graph type.
    
    Returns
    -------
    boolean: true, if the graph type is used for real-world graphs, else false.  
    """
    return graph_type.startswith("realworld-")

def split_graphs(all_graph_names, train_size= 0.6, valid_size=0.2, test_size=0.2):
    """Splits the graphs in a test, validate and test set.  
    
    Parameters
    ----------
    all_graph_names: array-like
        The graphs to split.
        
    train_size: float
        Percentage of graphs used for training.
        
    valid_size: float
        Percentage of graphs used for validation.
        
    test_size: float
        Percentage of graphs used for testing.
   
   Returns
    -------
    training set: array-like
        Graphnames used for training.
        
    validation set: array-like
        Graphnames used for validation.
        
    test set: array-like
        Graphnames used for test.   
    """
    if (train_size + valid_size + test_size > 1):
        print("No valid split")
    np.random.seed(constants.random_state)
    np.random.shuffle(all_graph_names)
    istances = len(all_graph_names)
    
    num_test_instances = int(istances * test_size)
    num_validate_instances = int(istances * valid_size)
    num_train_instances = int(istances * train_size)
    
    train, validate, test = np.split(
        all_graph_names,
        [
            num_train_instances,
            num_train_instances + num_validate_instances
        ]
    )
    train_list = train.tolist()
    validate_list = validate.tolist()
    test_list = test.tolist()
    
    if(train_size == 0.0):
        train_list = []
    if(valid_size == 0.0):
        validate_list = []
    if(test_size == 0.0):
        test_list = []
    return train_list, validate_list, test_list


def get_train_validate_test(data, graphs_for_enrichment, target, configurations):
    """Splits the data set into training, validation and test.
    
    Parameters
    ----------
    data: dataframe
        The dataframe with all available samples.
        
    graphs_for_enrichment: dataframe
        The dataframe with all available samples for enrichment.
        
    target: string
        The column name which should be predicted.
    
    configurations: array-like with cofigurations objects. e.g. [{"graph_type": [the graphtypes to consider], "training_size": 0.8, "validation_size": 0.2, "test_size": 0, enrich_by: 0.3}, ... , ]
        Is used to descibe which graph types to used for training/validation/test. For example, train and validate with R-MAT and test on real-wprld graphs and use 0.3 of the graphs of graphs_for_enrichment as enrichment.
    
    Returns
    ------- 
    training set: dataframe
        Samples used for training.
        
    validation set: dataframe
        Samples used for validation.
        
    test set: dataframe
        Samples used for test. 
        
    """
        
    columns = [column for column in data.columns if not column == target]
    graph_names_train = []
    graph_names_validate = []
    graph_names_test = []
    enrich_by = 0
    for configuration in configurations:
        if "enrich_by" in configuration.keys():
            enrich_by = configuration["enrich_by"]
            continue   
        for graph_type in configuration["graph_type"]:
            # Get graph names of the type
            graph_names_of_type = get_graph_names_of_graph_types(data, [graph_type])
            # Make stratified split per type
            _graphs_train, _graphs_validate, _graphs_test = split_graphs(
                graph_names_of_type,
                configuration["training_size"],
                configuration["validation_size"],
                configuration["test_size"])
            graph_names_train += _graphs_train
            graph_names_validate += _graphs_validate
            graph_names_test += _graphs_test     
    train_data = data[data['graph'].isin(graph_names_train)]
    X_train = train_data
    y_train = train_data[target]
    val_data = data[data['graph'].isin(graph_names_validate)]
    X_val = val_data
    y_val = val_data[target]
    test_data = data[data['graph'].isin(graph_names_test)]
    X_test = test_data
    y_test = test_data[target]
    print(f'Train data shape (before enrichment): {X_train.shape}')
    print(f'Validation data shape (before enrichment): {X_val.shape}')
    print(f'Test data shape (before enrichment): {X_test.shape}')
    if (enrich_by > 0):
        X_train_enriched, X_val_enriched, y_train_enriched, y_val_enriched = get_enrichment_data_set_train_validate(graphs_for_enrichment, enrich_by, target)
        X_train = pd.concat([X_train, X_train_enriched])
        X_val = pd.concat([X_val, X_val_enriched])
        y_train = pd.concat([y_train, y_train_enriched])
        y_val = pd.concat([y_val, y_val_enriched])
        print("Number of graphs for enrichment", len(X_train_enriched.graph.unique()) + len(X_val_enriched.graph.unique()) )
    print(f'Train data shape (after enrichment): {X_train.shape}')
    print(f'Validation data shape (after enrichment): {X_val.shape}')
    print(f'Test data shape (after enrichment): {X_test.shape}')
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_average_spearman(data_with_predictions, target):
    spearmans = []
    graphs = list(data_with_predictions.graph.unique())
    parts = list(data_with_predictions.num_partitions.unique())
    # Check the ranking of the partitioners given the graph and the number of partitioners.
    for g in graphs:
        for p in parts:
            filtered_data = data_with_predictions[
                (data_with_predictions.graph == g) &
                (data_with_predictions.num_partitions == p)
            ]
            if(len(filtered_data) > 0):
                spearman = stats.spearmanr(filtered_data[target], filtered_data["predicted_"+target])
                spearmans.append(spearman[0])
    return np.mean(spearmans)
    
def get_r2_rmse_mape_spearman_filtered_data(
    data_with_predictions, 
    graph_names = None, 
    graph_types = None, 
    num_partitions = None, 
    partitioner=None,
target=None):
    """ Calculate different scores for the predictions.
        The data can be filtered by partitioner, graphnames, number of partitions etc. to get the scores for different combinations
    
    Parameters
    ----------
    data_with_predictions: dataframe
        The samples with the predicted and actual replication factors. 
        
    graph_names: array-like
        Only consider smaples created out of the given names.
    graph_types: array-like
        Only consider smaples created out of graphs of the given graph types.
        
    num_partitions: array-like
        Only consider samples with partitions sizes contained in num_partitions.
    
    Returns
    -------
    r2 : float
    
    rmse: float
    
    mape: float
    
    spearman: float
    
    used_graph_names: array-like
        The used graphs.
        
    filtered_data: dataFrame
        The used data 
    
    """
    
    if graph_names is None:
        graph_names = list(data_with_predictions.graph.unique())
    if graph_types is None:
        graph_types = list(data_with_predictions.graph_type.unique())
    if num_partitions is None:
        num_partitions = list(data_with_predictions.num_partitions.unique())
    if partitioner is None:
        partitioner = list(data_with_predictions.partitioner.unique())
        
    filtered_data = data_with_predictions[
        (data_with_predictions["graph"].isin(graph_names)) & 
        (data_with_predictions["graph_type"].isin(graph_types)) & 
        (data_with_predictions["num_partitions"].isin(num_partitions)) & 
        (data_with_predictions["partitioner"].isin(partitioner))
    ]
    
    used_graph_names = list(filtered_data.graph.unique())
    used_graph_names.sort()
    
    r2 = metrics.r2_score(filtered_data[target], filtered_data["predicted_"+target])
    rmse = metrics.mean_squared_error(filtered_data[target], filtered_data["predicted_"+target], squared=False) 
    #mape = metrics.mean_absolute_percentage_error(filtered_data["replication_factor"], filtered_data["predicted_replication_factor"])
    y_true = filtered_data[target]
    y_pred = filtered_data["predicted_"+target]
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)
    m = np.average(output_errors)
    spearman = get_average_spearman(filtered_data, target)
    return r2, rmse, m, spearman, used_graph_names, filtered_data

    
def plot_feature_importance(importance_values, feature_names, level):
   # print(plot_feature_importance,importance_values, feature_names)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })
    
    def func(row):
        partitioner_names = []
        for partitioner in ['CRVC', 'HEP-100', 'HEP-10', '1DS', '1DD', '2D', 'DBH', '2PS', 'HDRF', 'HEP-1', 'NE', "HEP1","HEP10","HEP100"]:
            partitioner_names.append(partitioner)
            partitioner_names.append(partitioner.lower())
        
        
        if (row.Feature in partitioner_names):
            return "Partitioner"
        
        if (row.Feature == "num_partitions"):
            return "#Partitions"
        
        if (row.Feature == "pearson_mode_degrees_in"):
            return "In-Degree-Distribution"
        
        if (row.Feature == "pearson_mode_degrees_out"):
            return "Out-Degree-Distribution"
        
        if (row.Feature == "mean_degree"):
            return "Average Degree"
        
        if (row.Feature == "density"):
            return "Density"
        
        if (row.Feature == ""):
            return ""
        
    
        else:
            return row.Feature
    importance_df['Category'] = importance_df.apply(lambda row: func(row), axis=1)

    importance_df_aggregated = importance_df.groupby(['Category']).agg({'Importance': np.sum}).reset_index().sort_values(by=["Importance"], ascending=False)
    importance_df.sort_values(by=["Importance"], ascending=False)
    labels = importance_df_aggregated["Category"]
    sizes = importance_df_aggregated["Importance"]
    
    fig, ax1 = plt.subplots()
    
    ax1.pie(sizes, 
           # explode=explode, 
            labels=labels, autopct='%1.1f%%',
            )
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig.tight_layout() 
    fig.savefig("../figures/feature-importance-" +str(level) + ".pdf",  borderaxespad=0, bbox_inches='tight')
    plt.show()   

def get_enrichment_data_set_train_validate(graphs_for_enrichment, level, target):   
    """Get enrichment data for training (80%) and validation (20%).
    
    Parameters
    ----------
    graphs_for_enrichment: dataframe
        All samples available for enrichment.
    leve: float
        How many percent of the available graphs to use for enrichment.
    
    Returns
    -------
    train_data: dataframe
        Trainings data (input).
        
    val_data:
        Validation data (input).
        
    y_train:
        Trainings data (output)
        
    y_val:
        alidation data (output).
        
    """
    
    holdout_graphs_for_enrichment = list(graphs_for_enrichment.graph.unique())
    np.random.seed(None)
    np.random.shuffle(holdout_graphs_for_enrichment)
    
    possible_graphs = len(holdout_graphs_for_enrichment)
    if (level < 1.0):
        possible_graphs = int(level * possible_graphs)
        
    train, validate, =  np.split(
        holdout_graphs_for_enrichment[0:possible_graphs],
        [
            int(0.8*possible_graphs)
        ]
    )
    train_data = graphs_for_enrichment[graphs_for_enrichment['graph'].isin(train)]
    y_train = train_data[target]
    val_data = graphs_for_enrichment[graphs_for_enrichment['graph'].isin(validate)]
    y_val = val_data[target]
   # print(f'Train data shape (get_enrichment_data_set_train_validate): {X_train.shape}')
  #  print(f'Validation data shape (get_enrichment_data_set_train_validate): {X_val.shape}')
    return train_data, val_data, y_train, y_val

    

def plot_actual_vs_predicted(data, ax, title, target, xy_label):
    ax.scatter(data["predicted_"+target], data[target], marker="x")
    max_value = np.max([data["predicted_"+target].max(), data[target].max()])
    ax.plot([0,max_value], [0, max_value],c="b", label="Perfect",)
    ax.set(title=title, ylabel="Actual " +xy_label, xlabel="Predicted " + xy_label, label="Perfect") 
    ax.grid()
    ax.legend()
    
def plot_scores_by_type_and_partitioner(data, featureset):
    raw = []   
    for i,d in data.groupby(["level", "enriched_by", "partitioner"], as_index=False):
        raw.append({
            "partitioner": i[2],
            "level": i[0],
            "enriched_by": i[1],
            "rmse_mean": np.mean(d["rmse"]), 
            "rmse_std": np.std(d["rmse"]),
            "r2_mean": np.mean(d["r2"]), 
            "r2_std": np.std(d["r2"]),
            "mape_mean": np.mean(d["mape"]), 
            "mape_std": np.std(d["mape"]),
            "spearman_mean": np.mean(d["spearman"]), 
            "spearman_std": np.std(d["spearman"]),
        })

    scores_by_enrichment_aggreate_per_partitioner_type = pd.DataFrame(raw)
    test_level = list(scores_by_enrichment_aggreate_per_partitioner_type.level.unique())
    test_level = [x for x in test_level if not x == "all"] # by graphtype (level)
    enrichmend_levels = list(scores_by_enrichment_aggreate_per_partitioner_type.enriched_by.unique())
    enrichmend_levels.sort()
    plt.rcParams["figure.figsize"] = (8, 8)
    for l in enrichmend_levels:
        print("Enrichment", l)
        _data = scores_by_enrichment_aggreate_per_partitioner_type[scores_by_enrichment_aggreate_per_partitioner_type.enriched_by == l]
       # print(_data)
    #print("data for l", l,  _data)
    #pivot = test_results.pivot(index='test_level', columns='partitioner', values=['test_mape', "test_rmse"])
        _data["type"] = _data["level"].str.split("-").str[-1]
        _data = _data[~(_data.level == "all") & (~(_data.partitioner == "all")) ]
        pivot = _data.pivot(index="type", columns='partitioner', values='mape_mean')
        
      #  print(pivot.columns)
        #pivot.columns = [k for j,k in pivot.columns]
        #pivot.reset_index()
        
      #  print(pivot)
        fig,ax= plt.subplots()
        sn.heatmap(pivot, annot=True, cmap=sn.color_palette("Blues", as_cmap=True)
    )
        #plt.title("Enrichment by", str(l))
        plt.show()
        fig.savefig("../figures/heat-"+ featureset + str(l) + ".pdf",  borderaxespad=0, bbox_inches='tight')
        
def predict_and_cobine(X_test, model, target):
    data = X_test.copy()
    data["predicted_"+target] = model.predict(X_test)
    return data

# adapt.. not used
def plot_score_per_graph_type(data):
    d = data[data.description == description]
    fig, (ax1_r2, ax2_rmse, ax3_spearman) = plt.subplots(3, 1)
    fig.suptitle('Scores')
    ax1_r2.bar(d.graph_type, d.test_r2)
    ax2_rmse.bar(d.graph_type, d.test_rmse)
    ax3_spearman.bar(d.graph_type, d.spearman)

    ax1_r2.set_title("R2")
    ax2_rmse.set_title("RMSE")
    ax3_spearman.set_title("Spearman rank-order correlation coefficien")

    for ax in [ax1_r2, ax2_rmse, ax3_spearman]:
        ax.set_xlabel("Graph type")
        ax.tick_params(axis='x', rotation=45)
        ax.grid()
    fig.subplots_adjust(hspace=0.5)
    
    plt.show()
    

def get_random_forest(features, estimators, depths):
    """Creates Random Forest Pipeline.
    
    Parameters
    ----------
    features: array-like
        The features to consider.
    
    estimators:
        The number of estimators/trees in the forest.
        
    depths:
        The maximal depth the tree can reach.
    
    Returns
    -------
    pipline:
        The pipline incl. grid-search over the provided parameters.
        
    """
        
    random_forest_pipline = Pipeline(
        steps=[
            ('feature_selection', ColumnTransformer.ColumnTransformer()),
            ('scalar', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=constants.random_state))
        ]
    )
    
    random_forest_transformed = TransformedTargetRegressor(
        regressor=random_forest_pipline, 
        func=target_transform, 
        inverse_func=inverse_target_transform
    )

    random_forest_grid = {
        'regressor__feature_selection__features': features,
        'regressor__regressor__n_estimators': estimators,
        'regressor__regressor__max_depth': depths,
    }

    random_forest_model = GridSearchCV(
        random_forest_transformed, 
        random_forest_grid, 
        n_jobs=-1, 
        verbose=10, 
        cv=5)
    return random_forest_model

def train_random_forest(random_forest_model, data_train_validate_test, data_description, enrich_by, used_feature_set, target ):
    """Trains Random Forest Pipeline.
    
    Parameters
    ----------
    random_forest_model: pipeline
        The random forest pipeline.
    
    data_train_validate_test: array-like
        Data for training, validation and test. For example: [(X_train, X_val, X_test, y_train, y_val, y_test), ...]
    
    data_description: array-like
        The descriptions of data_train_validate_test. 
    
    enrich_by: array-like
        How much data_train_validate_test was enriched. 
        
    used_feature_set: array-like
        The features that are considered in data_train_validate_test.
        
    Returns
    -------
    validation_results: dataframe
        Scores on the validation set.
        
    test_results: dataframe
        Scores on the test set.
    """
        
    results_validation = []
    results_test = []
    #t_stamp = str(get_timestamp())
        
    for i in range(len(data_train_validate_test)):
        print("Train Random Forest - ", "Description:", data_description[i], "enriched by", enrich_by[i])
        X_train, X_val, X_test, y_train, y_val, y_test = data_train_validate_test[i]          
        random_forest_model.fit(X_train, y_train)
        data_with_predictions = predict_and_cobine(X_val, random_forest_model, target)
        # Scores on validation set.
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner= None,
            target=target)
        # Get best parameters.
        best_features = random_forest_model.best_params_["regressor__feature_selection__features"]
        best_num_estimators = random_forest_model.best_params_["regressor__regressor__n_estimators"]
        best_max_depth = random_forest_model.best_params_["regressor__regressor__max_depth"]  
        performance_validation = {
            'evaluation_set': "validation",
            'hyper_paramter_best_features': best_features,
            'hyper_paramter_best_num_estimators': best_num_estimators,
            'hyper_paramter_best_max_depth': best_max_depth,
            'description': data_description[i],
            'rmse': rmse,
            'mape':mape,
            'r2': r2 * 100, 
            'spearman': spearman,
            'enriched_by': enrich_by[i],
            "partitioner": "all"
        }
        
        results_validation.append(performance_validation)
        # Take the model with the best Hyper paramters 
        best_random_forest_model = random_forest_model.best_estimator_
        # Use Traing and Validatino set and retrain
        X_train_validate = pd.concat([X_train, X_val])
        y_train_validate = pd.concat([y_train, y_val])
        best_random_forest_model.fit(X_train_validate, y_train_validate)
        data_with_predictions = predict_and_cobine(X_test, best_random_forest_model, target)
        # Scores on test set.
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
        target=target)
        
        performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': "all",
            'rmse': rmse,
            'mape':mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all",
            "importance_values": best_random_forest_model.regressor_.named_steps['regressor'].feature_importances_,
            "importance_features":best_random_forest_model.regressor_.named_steps.feature_selection.features
        }
        
        results_test.append(performance_test)    
        all_graph_types = list(data_with_predictions.graph_type.unique())
        num_all_graph_types = len(all_graph_types)
        all_partitioner_strategies = list(data_with_predictions.partitioner.unique())
        num_all_partitioner_strategies = len(all_partitioner_strategies)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index] 
            for current_partitioner_index in range(num_all_partitioner_strategies):
                current_partitioner = all_partitioner_strategies[current_partitioner_index]
                r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=[current_partitioner],
                    target=target
            )
            
                performance_test = {
                'evaluation_set': "test",
                'description': data_description[i],
                'level': current_graph_types,
                'rmse': rmse,
                'mape': mape, 
                'r2': r2 * 100,
                'spearman': spearman,
                'partitioner': current_partitioner,
                'enriched_by':enrich_by[i],
                "importance_values": best_random_forest_model.regressor_.named_steps['regressor'].feature_importances_,
                "importance_features":best_random_forest_model.regressor_.named_steps.feature_selection.features
         }
                results_test.append(performance_test)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index]
            r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=None,
                target=target
            )
            
            performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': current_graph_types,
            'rmse': rmse,
            'mape':mape, 
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all",
             "importance_values": best_random_forest_model.regressor_.named_steps['regressor'].feature_importances_,
            "importance_features":best_random_forest_model.regressor_.named_steps.feature_selection.features
       
            }
            results_test.append(performance_test)
        
        if (STORE_RESULTS and (enrich_by[i] == 0.0 or enrich_by[i]==1.0)):   
            print("Store model...", "enrichment:", enrich_by[i])
            pickle.dump(best_random_forest_model, open(PATH_TO_STORE_MODELS + "/"+ get_time_stamp() + "_" + "RFR-" + str(used_feature_set) + "-" + str(enrich_by[i]) + "-"+target, 'wb'))
            #pickle.dump(best_random_forest_model, open("../models/RFR-" + str(used_feature_set) + "-" + str(enrich_by[i]) + "-" + t_stamp, 'wb'))
                                                 
    return pd.DataFrame(results_validation), pd.DataFrame(results_test)


def plot_scores_by_enrichment_level(data):
    scores = data[(data.partitioner=="all")]
    raw = []

    for i,d in scores.groupby(["level", "enriched_by"], as_index=False):
        raw.append({
            "level": i[0],
            "enriched_by": i[1],
            "rmse_mean": np.mean(d["rmse"]), 
            "rmse_std": np.std(d["rmse"]),
            "r2_mean": np.mean(d["r2"]), 
            "r2_std": np.std(d["r2"]),
            "mape_mean": np.mean(d["mape"]), 
            "mape_std": np.std(d["mape"]),
            "spearman_mean": np.mean(d["spearman"]), 
            "spearman_std": np.std(d["spearman"]),
        })
    scores_by_enrichment = pd.DataFrame(raw)   
    #print(scores_by_enrichment)
    linestyle_tuple = [
        ('x', (0, (5,1))),
        ('x', (0, (5,1,1,1))),
        ('x', (0, (5,1,1,1,1,1))),
        ('x', (0, (5,1,1,1,1,1,1,1))),
        ('x', (0, (10,1))),
        ('x', (0, (10,1,1,1))),
        ('x', (0, (10,1,1,1,1,1))),
        ('x', (0, (10,1,1,1,1,1,1,1))),
        ('x', (0, (1, 1))),
        ('x', (0, (20, 0))),
    ]
    c = -1
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (7,10)
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(4, 1)
    
    scores_by_enrichment["enriched_by_abs"] = scores_by_enrichment["enriched_by"] * 96
    
    print("MAPE wihout enrichment", scores_by_enrichment[(scores_by_enrichment.enriched_by == 0.0) & (scores_by_enrichment.level == "realworld-wiki")][["level","mape_mean"]])
    print("MAPE with 100% enrichment", scores_by_enrichment[(scores_by_enrichment.enriched_by == 1.0)& (scores_by_enrichment.level == "realworld-wiki")][["level","mape_mean"]])
    
    
    for i, d in scores_by_enrichment.groupby(["level"]):
       # print(d["level"].unique())
       # print("i", d)
        c += 1
        # marker="x"
        #print("pos", c%len(linestyle_tuple))
        ax[0].errorbar(d["enriched_by_abs"], d["mape_mean"], yerr = d["mape_std"], xerr = None, ls=linestyle_tuple[c%len(linestyle_tuple)][1], label=str(i)) 
        ax[1].errorbar(d["enriched_by_abs"], d["rmse_mean"], yerr = d["rmse_std"], xerr = None, ls=linestyle_tuple[c%len(linestyle_tuple)][1]) 
        ax[2].errorbar(d["enriched_by_abs"], d["r2_mean"], yerr = d["r2_std"], xerr = None, ls=linestyle_tuple[c%len(linestyle_tuple)][1] ) 
        ax[3].errorbar(d["enriched_by_abs"], d["spearman_mean"], yerr = d["spearman_std"], xerr = None, ls=linestyle_tuple[c%len(linestyle_tuple)][1]) 
        ax[0].set_xticks([0,19,38,57,76,96])
        ax[1].set_xticks([0,19,38,57,76,96])
        ax[2].set_xticks([0,19,38,57,76,96])
        ax[3].set_xticks([0,19,38,57,76,96])

    for a in ax:
        a.grid()

    ax[3].set_xlabel("# graphs")
    
    ax[0].set_ylabel("MAPE")
    ax[1].set_ylabel("RMSE")
    ax[2].set_ylabel("R$^2$")
    ax[3].set_ylabel("Spearman")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), markerscale=2., scatterpoints=1, handlelength=10, ncol=2)
    
   # fig.tight_layout() 
    plt.savefig("../figures/lines.pdf", borderaxespad=0, bbox_inches='tight')
    
    plt.show()

def get_knn(features, neighbors, ps):
    """Creates KNN Pipeline.
    
    Parameters
    ----------
    features: array-like
        The features to consider.
    
    neighbors:
        The number of neighbors to consider in KNN.
        
    ps:
        The exponent used in the distance metric.
    
    Returns
    -------
    pipline:
        The pipline incl. grid-search over the provided parameters.
        
    """
        
    knn_pipline = Pipeline(
        steps=[
            ('feature_selection', ColumnTransformer.ColumnTransformer()),
            ('scalar', StandardScaler()),
            ('regressor', KNeighborsRegressor(
                n_neighbors=3,
                weights="distance",
                algorithm="brute"
           )
            )
        ]
    )
    
    knn_transformed = TransformedTargetRegressor(
        regressor=knn_pipline, 
        func=target_transform, 
        inverse_func=inverse_target_transform
    )

    knn_grid = {
        'regressor__feature_selection__features': features,
       'regressor__regressor__n_neighbors': neighbors,
        'regressor__regressor__p': ps,
    }

    knn_model = GridSearchCV(
        knn_transformed, 
        knn_grid, 
        scoring=my_func,
        n_jobs=-1, 
        verbose=10, 
        cv=5)
    return knn_model

def train_knn(knn_model, data_train_validate_test, data_description, enrich_by, used_feature_set, target):
    """Trains KNN Pipeline.
    
    Parameters
    ----------
    knn_model: pipeline
        The KNN pipeline.
    
    data_train_validate_test: array-like
        Data for training, validation and test. For example: [(X_train, X_val, X_test, y_train, y_val, y_test), ...]
    
    data_description: array-like
        The descriptions of data_train_validate_test. 
    
    enrich_by: array-like
        How much data_train_validate_test was enriched. 
        
    used_feature_set: array-like
        The features that are considered in data_train_validate_test.
        
    Returns
    -------
    validation_results: dataframe
        Scores on the validation set.
        
    test_results: dataframe
        Scores on the test set.
    """
        
    results_validation = []
    results_test = []
    #t_stamp = str(get_timestamp())
        
    for i in range(len(data_train_validate_test)):
        print("Train KNN - ", "Description:", data_description[i], "enriched by", enrich_by[i])
        X_train, X_val, X_test, y_train, y_val, y_test = data_train_validate_test[i]      
        X_train_validate = pd.concat([X_train, X_val])
        y_train_validate = pd.concat([y_train, y_val])    



        # Train 1
        knn_model.fit(X_train_validate, y_train_validate)
        data_with_predictions = predict_and_cobine(X_train_validate, knn_model, target)
        # Validate with validation data
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
            target=target)
        
        hyper_params_neighbors = knn_model.best_params_['regressor__regressor__n_neighbors']
        hyper_params_p= knn_model.best_params_['regressor__regressor__p']
        hyper_params_features = knn_model.best_params_["regressor__feature_selection__features"]

        performance_validation = {
            'evaluation_set': "validation",
            'hyper_parameter_best_neighbors': hyper_params_neighbors,
            'hyper_parameter_best_p': hyper_params_p,
            'hyper_paramter_best_features': hyper_params_features,
            'description': data_description[i],
            'rmse': rmse,
            'mape':knn_model.cv_results_["mean_test_score"][knn_model.best_index_],
            'r2': r2 * 100,
          #  "cv-score": knn_model.cv_results_["mean_test_score"],
            'spearman': spearman,
            'enriched_by': enrich_by[i],
            "partitioner": "all"
        }
        
        results_validation.append(performance_validation)
         # Take the model with the best Hyper paramters 
       



       # best_knn_model = knn_model.best_estimator_
        # Use Traing and Validatino set and Retrain
        #X_train_validate = pd.concat([X_train, X_val])
        #y_train_validate = pd.concat([y_train, y_val])
        
        #best_knn_model.fit(X_train_validate, y_train_validate)





        
        data_with_predictions = predict_and_cobine(X_test, knn_model, target)
        
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
        target=target)
        
        performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': "all",
            'rmse': rmse,
            'mape': mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
        }
        
        results_test.append(performance_test)
        
        all_graph_types = list(data_with_predictions.graph_type.unique())
        num_all_graph_types = len(all_graph_types)
 
        all_partitioner_strategies = list(data_with_predictions.partitioner.unique())
        num_all_partitioner_strategies = len(all_partitioner_strategies)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index] 
            for current_partitioner_index in range(num_all_partitioner_strategies):
                current_partitioner = all_partitioner_strategies[current_partitioner_index]
                r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=[current_partitioner],
                target=target
            )
            
                performance_test = {
                'evaluation_set': "test",
                'description': data_description[i],
                'level': current_graph_types,
                'rmse': rmse,
                'mape':mape, 
                'r2': r2 * 100,
                'spearman': spearman,
                'partitioner': current_partitioner,
                'enriched_by':enrich_by[i]
                }
                results_test.append(performance_test)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index]
            r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=None,
                target=target
            )
            
            performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': current_graph_types,
            'rmse': rmse,
            'mape':mape, 
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
            }
            results_test.append(performance_test)
    
        
        if (STORE_RESULTS and (enrich_by[i] == 0.0 or enrich_by[i]==1.0)):   
            print("Store model...", "enrichment:", enrich_by[i])
            pickle.dump(knn_model, open(PATH_TO_STORE_MODELS +"/"+ get_time_stamp() + "_" +  "KNN-" + str(used_feature_set) + "-" + str(enrich_by[i])+ "-"+target, 'wb'))
 

                                                        
    return pd.DataFrame(results_validation), pd.DataFrame(results_test)



    
def get_svr(features=None, kernels=None, interval_C=None,  interval_epsilon=None, interval_gamma=None):
    """Creates SVR Pipeline.
    
    Parameters
    ----------
    features: array-like
        The features to consider.

    kernals: array-like
        The kernals to consider.
    
    interval_C: array-like
        C regularization parametera.
        Near the maximum value of the label [Foundations of Machine Learning]
        
    interval_epsilon: array-like
        Epsilon in the epsilon-SVR model. 
        Average distance of labels can be used [Foundations of Machine Learning]

    interval_gamma: array-like
        The kernal coefficient is only considered for the kernals: rbf, poly, sigmoid
    
    Returns
    -------
    pipline:
        The pipline incl. grid-search over the provided parameters.
        
    """
 
    svr_pipline = Pipeline(
        steps=[
            ('feature_selection', ColumnTransformer.ColumnTransformer(features)),
            ('scalar', StandardScaler()),
            ('regressor', SVR())
        ]
    )
    
    print(svr_pipline.get_params().keys())


    svr_transformed = TransformedTargetRegressor(
        regressor=svr_pipline, 
        func=target_transform, 
        inverse_func=inverse_target_transform
    )

    svr_grid = {}

    if not kernels is None:
        svr_grid['regressor__regressor__kernel'] = kernels

    if not interval_C is None:
        svr_grid['regressor__regressor__C'] = interval_C

    if not interval_epsilon is None:
        svr_grid['regressor__regressor__epsilon'] = interval_epsilon

    if not interval_gamma is None:
        svr_grid['regressor__regressor__gamma'] = interval_gamma
  
    svr_model = GridSearchCV(
        svr_transformed, 
        svr_grid, 
        n_jobs=-1, 
        verbose=10, 
        cv=5,
       # scoring = "neg_mean_absolute_error"
        )
    return svr_model

def train_svr(svr_model, data_train_validate_test, data_description, enrich_by, used_feature_set, target):
    """Trains SVR Pipeline.
    
    Parameters
    ----------
    svr_model: pipeline
        The SVR pipeline.
    
    data_train_validate_test: array-like
        Data for training, validation and test. For example: [(X_train, X_val, X_test, y_train, y_val, y_test), ...]
    
    data_description: array-like
        The descriptions of data_train_validate_test. 
    
    enrich_by: array-like
        How much data_train_validate_test was enriched. 
        
    used_feature_set: array-like
        The features that are considered in data_train_validate_test.
        
    Returns
    -------
    validation_results: dataframe
        Scores on the validation set.
        
    test_results: dataframe
        Scores on the test set.
    """
        
    results_validation = []
    results_test = []
    #t_stamp = str(get_timestamp())
        
    for i in range(len(data_train_validate_test)):
        print("{}/{}".format(i, len(data_train_validate_test)))
        print("Train SVR - ", "Description:", data_description[i], "enriched by", enrich_by[i])
        X_train, X_val, X_test, y_train, y_val, y_test = data_train_validate_test[i]          
        svr_model.fit(X_train, y_train)
        data_with_predictions = predict_and_cobine(X_val, svr_model, target)
        # Validate with validation data
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
            target=target)
     
        print("Results: ", svr_model.cv_results_)

        print("Results: dir", dir(svr_model.cv_results_))
        print("Results keys: ", svr_model.cv_results_.keys())
        print("Results values: ", svr_model.cv_results_.values())

        print("Results values df: ", pd.DataFrame(svr_model.cv_results_.values()))

 

        result = pd.concat([pd.DataFrame(svr_model.cv_results_["params"]), pd.DataFrame({"score": svr_model.cv_results_["mean_test_score"]})], axis=1)


        result.to_csv("../models/SVR_{}_{}_{}-crossvalidation.csv".format(used_feature_set, enrich_by[i],  target))
        print("Results Scores: ",result)



        hyperparamter_best_C = svr_model.get_params()["estimator__regressor__regressor__C"]
        hyperparamter_best_kernel = svr_model.get_params()["estimator__regressor__regressor__kernel"]
        hyperparamter_best_epsilon = svr_model.get_params()["estimator__regressor__regressor__epsilon"]
        hyperparamter_best_gamma = svr_model.get_params()["estimator__regressor__regressor__gamma"]

        field = "regressor__regressor__C"
        if field in svr_model.best_params_:
            hyperparamter_best_C = svr_model.best_params_[field]

        field = "regressor__regressor__kernel"
        if field in svr_model.best_params_:
            hyperparamter_best_kernel = svr_model.best_params_[field]

        field = "regressor__regressor__epsilon"
        if field in svr_model.best_params_:
            hyperparamter_best_epsilon = svr_model.best_params_[field]
        
        field = "regressor__regressor__gamma"
        if field in svr_model.best_params_:
            hyperparamter_best_gamma = svr_model.best_params_[field]
        
        print("Best Params", svr_model.best_params_)

        performance_validation = {
            'evaluation_set': "validation",
            'hyperparamter_best_C': hyperparamter_best_C,
            'hyperparamter_best_kernel': hyperparamter_best_kernel,
            'hyperparamter_best_epsilon': hyperparamter_best_epsilon,
            'hyperparamter_best_gamma': hyperparamter_best_gamma,
           # 'hyper_paramter_best_features': hyper_params_features,
            'description': data_description[i],
            'rmse': rmse,
            'mape':mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by': enrich_by[i],
            "partitioner": "all"
        }
        
        results_validation.append(performance_validation)
         # Take the model with the best Hyper paramters 
        print("best svr model", svr_model)
        best_svr_model = svr_model.best_estimator_
        # Use Traing and Validatino set and Retrain
        X_train_validate = pd.concat([X_train, X_val])
        y_train_validate = pd.concat([y_train, y_val])
        
        best_svr_model.fit(X_train_validate, y_train_validate)
        
        data_with_predictions = predict_and_cobine(X_test, best_svr_model, target)
        
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
        target=target)
        
        performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': "all",
            'rmse': rmse,
            'mape': mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
        }
        
        results_test.append(performance_test)
        
        all_graph_types = list(data_with_predictions.graph_type.unique())
        num_all_graph_types = len(all_graph_types)
 
        all_partitioner_strategies = list(data_with_predictions.partitioner.unique())
        num_all_partitioner_strategies = len(all_partitioner_strategies)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index] 
            for current_partitioner_index in range(num_all_partitioner_strategies):
                current_partitioner = all_partitioner_strategies[current_partitioner_index]
                r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=[current_partitioner],
                target=target
            )
            
                performance_test = {
                'evaluation_set': "test",
                'description': data_description[i],
                'level': current_graph_types,
                'rmse': rmse,
                'mape':mape, 
                'r2': r2 * 100,
                'spearman': spearman,
                'partitioner': current_partitioner,
                'enriched_by':enrich_by[i]
                }
                results_test.append(performance_test)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index]
            r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=None,
                target=target
            )
            
            performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': current_graph_types,
            'rmse': rmse,
            'mape':mape, 
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
            }
            results_test.append(performance_test)
    
        
        if (STORE_RESULTS and (enrich_by[i] == 0.0 or enrich_by[i]==1.0)):   
            print("Store model...", "enrichment:", enrich_by[i])
            pickle.dump(best_svr_model, open(PATH_TO_STORE_MODELS + "/"+ get_time_stamp() + "_" +"SVR-" + str(used_feature_set) + "-" + str(enrich_by[i])+ "-"+target, 'wb'))
 
                                                        
    return pd.DataFrame(results_validation), pd.DataFrame(results_test)


      
def get_xgb(features=None, kernels=None, interval_C=None,  interval_epsilon=None, interval_gamma=None):
    """Creates SVR Pipeline.
    
    Parameters
    ----------
    features: array-like
        The features to consider.

    kernals: array-like
        The kernals to consider.
    
    interval_C: array-like
        C regularization parametera.
        Near the maximum value of the label [Foundations of Machine Learning]
        
    interval_epsilon: array-like
        Epsilon in the epsilon-SVR model. 
        Average distance of labels can be used [Foundations of Machine Learning]

    interval_gamma: array-like
        The kernal coefficient is only considered for the kernals: rbf, poly, sigmoid
    
    Returns
    -------
    pipline:
        The pipline incl. grid-search over the provided parameters.
        
    """
 
    svr_pipline = Pipeline(
        steps=[
            ('feature_selection', ColumnTransformer.ColumnTransformer(features)),
            ('scalar', StandardScaler()),
            ('regressor',  XGBRegressor(n_estimators=800, max_depth=15, objective='reg:squarederror'))
        ]
    )
    
    print(svr_pipline.get_params().keys())


    svr_transformed = TransformedTargetRegressor(
        regressor=svr_pipline, 
        func=target_transform, 
        inverse_func=inverse_target_transform
    )

    svr_grid = {
        "regressor__regressor__n_estimators":[800],
        "regressor__regressor__max_depth":[15],
        "regressor__regressor__objective":['reg:squarederror'],
        }

    

    if not kernels is None:
        svr_grid['regressor__regressor__kernel'] = kernels

    if not interval_C is None:
        svr_grid['regressor__regressor__C'] = interval_C

    if not interval_epsilon is None:
        svr_grid['regressor__regressor__epsilon'] = interval_epsilon

    if not interval_gamma is None:
        svr_grid['regressor__regressor__gamma'] = interval_gamma
  
    svr_model = GridSearchCV(
        svr_transformed, 
        svr_grid, 
        n_jobs=-1, 
        verbose=10, 
        cv=2,
       # scoring = "neg_mean_absolute_error"
        )
    return svr_model

def train_xgb(svr_model, data_train_validate_test, data_description, enrich_by, used_feature_set, target):
    """Trains SVR Pipeline.
    
    Parameters
    ----------
    svr_model: pipeline
        The SVR pipeline.
    
    data_train_validate_test: array-like
        Data for training, validation and test. For example: [(X_train, X_val, X_test, y_train, y_val, y_test), ...]
    
    data_description: array-like
        The descriptions of data_train_validate_test. 
    
    enrich_by: array-like
        How much data_train_validate_test was enriched. 
        
    used_feature_set: array-like
        The features that are considered in data_train_validate_test.
        
    Returns
    -------
    validation_results: dataframe
        Scores on the validation set.
        
    test_results: dataframe
        Scores on the test set.
    """
        
    results_validation = []
    results_test = []
    #t_stamp = str(get_timestamp())
        
    for i in range(len(data_train_validate_test)):
        print("{}/{}".format(i, len(data_train_validate_test)))
        print("Train XGB - ", "Description:", data_description[i], "enriched by", enrich_by[i])
        X_train, X_val, X_test, y_train, y_val, y_test = data_train_validate_test[i]          
        svr_model.fit(X_train, y_train)
        data_with_predictions = predict_and_cobine(X_val, svr_model, target)
        # Validate with validation data
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
            target=target)
     
        print("Results: ", svr_model.cv_results_)

        print("Results: dir", dir(svr_model.cv_results_))
        print("Results keys: ", svr_model.cv_results_.keys())
        print("Results values: ", svr_model.cv_results_.values())

        print("Results values df: ", pd.DataFrame(svr_model.cv_results_.values()))

 

        result = pd.concat([pd.DataFrame(svr_model.cv_results_["params"]), pd.DataFrame({"score": svr_model.cv_results_["mean_test_score"]})], axis=1)


        result.to_csv("../models/XGB_{}_{}_{}-crossvalidation.csv".format(used_feature_set, enrich_by[i],  target))
        print("Results Scores: ",result)



        hyperparamter_best_C = 0# svr_model.get_params()["estimator__regressor__regressor__C"]
        hyperparamter_best_kernel = 0#svr_model.get_params()["estimator__regressor__regressor__kernel"]
        hyperparamter_best_epsilon =0# svr_model.get_params()["estimator__regressor__regressor__epsilon"]
        hyperparamter_best_gamma = 0#svr_model.get_params()["estimator__regressor__regressor__gamma"]

        field = "regressor__regressor__C"
        if field in svr_model.best_params_:
            hyperparamter_best_C = svr_model.best_params_[field]

        field = "regressor__regressor__kernel"
        if field in svr_model.best_params_:
            hyperparamter_best_kernel = svr_model.best_params_[field]

        field = "regressor__regressor__epsilon"
        if field in svr_model.best_params_:
            hyperparamter_best_epsilon = svr_model.best_params_[field]
        
        field = "regressor__regressor__gamma"
        if field in svr_model.best_params_:
            hyperparamter_best_gamma = svr_model.best_params_[field]
        
        print("Best Params", svr_model.best_params_)

        performance_validation = {
            'evaluation_set': "validation",
            'hyperparamter_best_C': hyperparamter_best_C,
            'hyperparamter_best_kernel': hyperparamter_best_kernel,
            'hyperparamter_best_epsilon': hyperparamter_best_epsilon,
            'hyperparamter_best_gamma': hyperparamter_best_gamma,
           # 'hyper_paramter_best_features': hyper_params_features,
            'description': data_description[i],
            'rmse': rmse,
            'mape':mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by': enrich_by[i],
            "partitioner": "all"
        }
        
        results_validation.append(performance_validation)
         # Take the model with the best Hyper paramters 
        print("best svr model", svr_model)
        best_svr_model = svr_model.best_estimator_
        # Use Traing and Validatino set and Retrain
        X_train_validate = pd.concat([X_train, X_val])
        y_train_validate = pd.concat([y_train, y_val])
        
        best_svr_model.fit(X_train_validate, y_train_validate)
        
        data_with_predictions = predict_and_cobine(X_test, best_svr_model, target)
        
        r2, rmse, mape, spearman, graphs_used_for_train, filtered_data = get_r2_rmse_mape_spearman_filtered_data(
            data_with_predictions, 
            graph_names = None, 
            graph_types = None, 
            num_partitions = None, 
            partitioner=None,
        target=target)
        
        performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': "all",
            'rmse': rmse,
            'mape': mape,
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
        }
        
        results_test.append(performance_test)
        
        all_graph_types = list(data_with_predictions.graph_type.unique())
        num_all_graph_types = len(all_graph_types)
 
        all_partitioner_strategies = list(data_with_predictions.partitioner.unique())
        num_all_partitioner_strategies = len(all_partitioner_strategies)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index] 
            for current_partitioner_index in range(num_all_partitioner_strategies):
                current_partitioner = all_partitioner_strategies[current_partitioner_index]
                r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=[current_partitioner],
                target=target
            )
            
                performance_test = {
                'evaluation_set': "test",
                'description': data_description[i],
                'level': current_graph_types,
                'rmse': rmse,
                'mape':mape, 
                'r2': r2 * 100,
                'spearman': spearman,
                'partitioner': current_partitioner,
                'enriched_by':enrich_by[i]
                }
                results_test.append(performance_test)
        
        for graph_type_index in range(num_all_graph_types):  
            current_graph_types = all_graph_types[graph_type_index]
            r2, rmse, mape, spearman, graphs_used_for_train, filtered_data  = get_r2_rmse_mape_spearman_filtered_data(
                data_with_predictions, 
                graph_names = None, 
                graph_types = [current_graph_types], 
                num_partitions = None, 
                partitioner=None,
                target=target
            )
            
            performance_test = {
            'evaluation_set': "test",
            'description': data_description[i],
            'level': current_graph_types,
            'rmse': rmse,
            'mape':mape, 
            'r2': r2 * 100,
            'spearman': spearman,
            'enriched_by':enrich_by[i],
            "partitioner": "all"
            }
            results_test.append(performance_test)
    
        
        if (STORE_RESULTS and (enrich_by[i] == 0.0 or enrich_by[i]==1.0)):   
            print("Store model...", "enrichment:", enrich_by[i])
            pickle.dump(best_svr_model, open(PATH_TO_STORE_MODELS + "/"+ get_time_stamp() + "_" + "XGB-" + str(used_feature_set) + "-" + str(enrich_by[i])+ "-"+target, 'wb'))
 
                                                        
    return pd.DataFrame(results_validation), pd.DataFrame(results_test)