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
class Statistic():
    def __init__(self, values):
        self.mean = np.mean(values)
        self.p25 = np.percentile(values, 25)
        self.p50 = np.percentile(values, 50)
        self.p75 = np.percentile(values, 75)
        self.max = np.max(values)
        
    def log(self):
        print(self.as_string())
        
    def as_string(self):
        return "Mean: {} 25%: {} 50%: {} 75%: {} Max: {}".format(self.mean, self.p25, self.p50, self.p75, self.max)

class RankStatistics():
    def __init__(self, values_expected_ordering, values_actual_ordering):
        self.spearman = stats.spearmanr(values_expected_ordering, values_actual_ordering)[0]
        
def overview_target_distribution(targets):
    """
    For each target get aggregated values by dataset and partitioner. 
    """
    results = []
    types = ["rmat-small","rmat-medium", "realworld-", "barabasi"]
    partitioner = combined_graphs_encoded.partitioner.unique()
    for target in targets:
        for t in types:
            for p in partitioner:
                data = combined_graphs_encoded[
                    (combined_graphs_encoded.graph_type.str.startswith(t)) &
                    (combined_graphs_encoded.partitioner == p)
                ]
                values = data[target].to_numpy()
                statistics = Statistic(values)
                results.append({
                    "dataset":t,
                    "target":target,
                    "partitioner": p,
                    "mean":statistics.mean,
                    "25":statistics.p25,
                    "50":statistics.p50,
                    "75":statistics.p75,
                    "max":statistics.max
                })
    return pd.DataFrame(results).sort_values(by=["target", "dataset", "partitioner"])

def overview_target_rank_order(targets):
    """
    For each graph and partitioner we sort by number of partitions.
    They question to answer is whether the target metric (e.g. the replication factor or a balance metric) increases
    if we increase the number of partitions. For the replication factor that more or less always the case.
    """
    result = []
    for target in targets:
        for index, data in combined_graphs_encoded.groupby(["graph", "partitioner"]):
            values_actual_ordering = data.sort_values(by=["num_partitions"], ascending=True)[target].to_numpy()
            values_expected_ordering = np.sort(values_actual_ordering)
            rs = RankStatistics(values_expected_ordering, values_actual_ordering)
            if not np.isnan(rs.spearman):
                result.append({
                    "graph": index[0],
                    "target": target,
                    "partitioner": index[1],
                    "spearman": rs.spearman
                })
    result_df = pd.DataFrame(result)
    for index, data in result_df.groupby(["target"]):
        print(index, Statistic(data.spearman.to_numpy()).as_string())
        
def show_graph_for_partitioner(graph, partition_sizes,  partitioners):
    targets = [
        'replication_factor',
        'edge_balance', 
        'vertex_balance', 
        'destination_balance',
        'source_balance', 
    ]
    result = combined_graphs_encoded[
        (combined_graphs_encoded.graph == graph) &
        (combined_graphs_encoded.partitioner.isin(partitioners)&
        (combined_graphs_encoded.num_partitions.isin(partition_sizes))) 
    ][targets + ["num_partitions", "partitioner", "num_edges", "origin"]].sort_values(by=["num_partitions"]) 
    return result.sort_values(by=["replication_factor"])

def mape_of_df(df, actual, predicted):
    df["mape"] = np.abs(df[actual] - df[predicted]) / df[actual]
    return df


def plot_scores_by_enrichment_level(data, target):
    scores = data[(data.partitioner == "all")]
    raw = []

    for i,d in scores.groupby(["level", "enriched_by"], as_index=False):
        raw.append({
            "level": i[0],
            "enriched_by": i[1],
            "rmse_mean": np.mean(d["rmse"]), 
            "rmse_std": np.std(d["rmse"]),
            "r2_mean": np.mean(d["r2"]/100), 
            "r2_std": np.std(d["r2"]/100),
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
  #  plt.rcParams["figure.figsize"] = (14,30)
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(4, 1)
    
    scores_by_enrichment["enriched_by_abs"] = scores_by_enrichment["enriched_by"] * 96
    
    print("MAPE wihout enrichment", scores_by_enrichment[(scores_by_enrichment.enriched_by == 0.0) & (scores_by_enrichment.level == "realworld-wiki")][["level","mape_mean"]])
    print("MAPE wih 20% enrichment", scores_by_enrichment[(scores_by_enrichment.enriched_by == 0.2) & (scores_by_enrichment.level == "realworld-wiki")][["level","mape_mean"]])
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
    plt.savefig("../figures/{}_lines.pdf".format(target), borderaxespad=0, bbox_inches='tight')
    
    plt.show()



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
    
    
    
    scores_by_enrichment_aggreate_per_partitioner_type = pd.DataFrame(raw).round(2)
    max_val = scores_by_enrichment_aggreate_per_partitioner_type[["mape_mean"]].max()
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
       # print(_data)
        pivot = _data.pivot(index="type", columns='partitioner', values='mape_mean')

      #  print(pivot.columns)
        #pivot.columns = [k for j,k in pivot.columns]
        #pivot.reset_index()

      #  print(pivot)
        fig,ax= plt.subplots()
        sn.heatmap(pivot, annot=True, cmap=sn.color_palette("Blues", as_cmap=True),  vmin=0, vmax=1)
        #plt.title("Enrichment by", str(l))
        plt.show()
        fig.savefig("../figures/heat_"+  featureset + "_" + str(l) + ".pdf",  borderaxespad=0, bbox_inches='tight')

    
    if False:
        # Only to show what the difference is in terms of difference mape with enrichment 0 and enrichment 1  
        colors = [
        (0, "#ff0303"), 
        #(0.3, "#ff6200"), 
        #(0.5, "#4dc46b"),
        (1.0, "#00ff22")]

        colors = [
        (0, "#ff001e"), 
        #(0.3, "#ff6200"), 
        #(0.5, "#4dc46b"),
        (2.0, "#0059ff")]

        colors = [
        (0, "#0059ff"), 
        #(0.3, "#ff6200"), 
        #(0.5, "#4dc46b"),
        (1.0, "#ff001e")]

        from matplotlib.colors import LinearSegmentedColormap
        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom_navy',
            colors=colors,
        )
        _data_0 = scores_by_enrichment_aggreate_per_partitioner_type[scores_by_enrichment_aggreate_per_partitioner_type.enriched_by == 0.0]
        _data_1 = scores_by_enrichment_aggreate_per_partitioner_type[scores_by_enrichment_aggreate_per_partitioner_type.enriched_by == 1.0]

        _data_0["type"] = _data_0["level"].str.split("-").str[-1]
        _data_1["type"] = _data_1["level"].str.split("-").str[-1]

        _data_0 = _data_0[~(_data_0.level == "all") & (~(_data_0.partitioner == "all")) ] 
        _data_1 = _data_1[~(_data_1.level == "all") & (~(_data_1.partitioner == "all")) ] 

        _data = pd.merge(_data_0,_data_1, on=["partitioner", "level"] )
        #print(_data)
        _data["mape_mean"] = (_data["mape_mean_y"] - _data["mape_mean_x"])

        pivot = _data.pivot(index="type_x", columns='partitioner', values='mape_mean')

        fig,ax= plt.subplots()
        sn.heatmap(pivot, annot=True, cmap=custom_color_map, vmax=0.5, vmin=-0.5,)
        #plt.title("Enrichment by", str(l))
        plt.show()
        fig.savefig("../figures/0-vs-1-enrichment_heat_"+  featureset + "_" + str(l) + ".pdf",  borderaxespad=0, bbox_inches='tight')



