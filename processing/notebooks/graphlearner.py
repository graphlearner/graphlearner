import argparse
import pandas as pd
import numpy as np
import pickle as pic
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--graph_properties', metavar="",  type=str, required=True, help="Relative path to the file which contains the graph properties." )
parser.add_argument('--partitioners',metavar="",  nargs='+', type=str, required=False, default=['crvc','hep100','hep10','1ds','1dd','2d','dbh','2ps','hdrf','hep1','ne'], help="The partitioners for which graphlearner should be applied. Default:all, Options: dbh, 2ps, hdrf, crvc, hep100, hep10, 1ds, 1dd, 2d, hep1, ne")
parser.add_argument('--num_partitions',metavar="",  type=int, required=True, help="The number of partitions." )
parser.add_argument('--processing_algorithm', metavar="", type=str, required=False, default="cc", help="The graph processing algorithm for which a partitioner should be selected. Default: cc, Options: pr, cc, sssp, k-cores, syn-low, syn-high")
parser.add_argument('--num_iterations',metavar="",  type=int, required=True, help="The number of iterations. Only used for pr, synthetic-low and synthetic-high" )

# Set the paths where the models are stored
BASE_PATH_PARTITIONING_RUNTIME_MODELS = "/home/ubuntu/cephstorage/partitioning-run-time-prediction-models"
BASE_PATH_PROCESSING_RUNTIME_MODELS = "/home/ubuntu/cephstorage/graph-processing-run-time-prediction-models"
BASE_PATH_QUALITY_MODELS = "/home/ubuntu/cephstorage/projects/graphlearner/quality/serialized_models"
args, unknown = parser.parse_known_args()

all_partitioners = ['crvc','hep100','hep10','1ds','1dd','2d','dbh','2ps','hdrf','hep1','ne']

selected_partitioners = args.partitioners
num_parts = args.num_partitions
metrics_files = args.graph_properties
algorithm = args.processing_algorithm
num_iterations = args.num_iterations

class GraphLearner():
    """ This class represents GraphLearner. GraphLearner consists of four components: 
    1. PartitioningQualityPredictor, to predict the partitioning quality metrics (see method: PartitioningQualityPredictor)
    2. PartitioningTimePredictor, to predict the partitioning run-time (see method: PartitioningTimePredictor)
    3. ProcessingTimePredictor, to predict the graph processing run-time Based on these three predictions (see method: ProcessingTimePredictor)
    4. PartitionerSelector, to select the partitioner that either minimizes the graph processing or the end-to-end run-time. (see method: PartitionerSelector)
    """

    def load_partitioning_time(self, timestamp,  best_model_name):
        """Helper to load the serialized model to predict the partitioning run-time.

        Args:
            timestamp (str): The timestamp, needed to identify the serialized model
            best_model_name (str): the model name like for example XGB, needed to identify the serialized model

        Returns:
            model: the model to predict partitioning run-times. 
        """
        paths_to_models = glob.glob("{}/{}/*".format(BASE_PATH_PARTITIONING_RUNTIME_MODELS, timestamp))
        # Filter the models 
        filtered = list(filter(lambda paths_to_model: "single-model-for-all-partitioner" in paths_to_model and best_model_name in paths_to_model , paths_to_models))
        # Verify the we were able to select the model.
        if len(filtered) != 1:
            print(filtered)
            print("We have a problem. We have NOT selected one model.")
        path_to_model = filtered[0]
        model = pic.load(open(path_to_model, 'rb'))
        return model

    def load_processing_time(self, timestamp, best_model_name, algorithm):
        """Helper to load the model to predict the processing run-time for the given algorithm.

        Args:
            timestamp (str): The timestamp, needed to identify the serialized model
            best_model_name (str): the model name like for example XGB, needed to identify the serialized model
            algorithm (str): The graph processing algorithm for which to predict the processing time. E.g., pr, cc, ...

        Returns:
            model: the model to predict processing run-time for the given graph processing algorithm. 
        """
        algorithm_pattern = "_{}_".format(algorithm)
        paths_to_models = glob.glob("{}/{}/*".format(BASE_PATH_PROCESSING_RUNTIME_MODELS, timestamp))
        filtered = list(filter(lambda paths_to_model: ("single-model-for-all-partitioner" in paths_to_model) and (best_model_name in paths_to_model) and algorithm_pattern in paths_to_model , paths_to_models))
        if len(filtered) != 1:
            print(filtered)
            print("We have a problem. We have NOT selected one model. ")
        path_to_model = filtered[0]
        model = pic.load(open(path_to_model, 'rb'))
        return model
    
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
            partitioning_model_name (str): The model for the partitioning run-time prediction. E.g., XGB
            partitioning_model_timestamp (str): The timestamp, needed to identify the serialized model
            processing_model_description list(str, str): For which graph processing should GraphLearner select which model. e.g, [("pr", "XGB"), ("cc", "SVR")]
            processing_model_timestamp (str): The timestamp, needed to identify the serialized model
            quality_model_description (str): We trained the graph quality predictor with two feature sets: "Easy" and "Hard".
            number_iterations (int): Will be set to one for cc, sssp, kcore as we predict the time until convergence. For the algorithms pr, synthetic-low and synthetic-high the number of iterations needs to be set. See Section 4.4
        """

        # Load serialized models
        self.partitioning_runtime_model = self.load_partitioning_time(timestamp=partitioning_model_timestamp, best_model_name=partitioning_model_name)
        self.processing_runtime_models = {}
        self.quality_models = {}

        def get_quality_model_path(timestamp, model_name, featureset, enrichment_level, target):
            return BASE_PATH_QUALITY_MODELS + "/{}_{}-{}-{}-{}".format(timestamp, model_name, featureset, enrichment_level, target)
        
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
        for description in processing_model_description:
            algorithm = description["algorithm"]
            model_name = description["model_name"]
            self.processing_runtime_models[algorithm] = self.load_processing_time(timestamp=processing_model_timestamp, best_model_name=model_name, algorithm=algorithm)
    
    def PartitioningQualityPredictor(self, data):
        _data = data.copy() 
        _data["replication_factor"] = self.quality_models["replication_factor"].predict(_data)
        _data["edge_balance"] = self.quality_models["edge_balance"].predict(_data)
        _data["source_balance"] = self.quality_models["source_balance"].predict(_data)
        _data["destination_balance"] = self.quality_models["destination_balance"].predict(_data)
        _data["vertex_balance"] = self.quality_models["vertex_balance"].predict(_data)
        return _data

    def set_number_iterations(self, number_iterations):
        self.num_iterations = number_iterations

    def get_num_iterations(self, algorithm):
        """ As described in Section 4.4 for some graph processing algorithms we the number of iterations needs to be set.  

        Args:
            algorithm (str): The graph processing algorithm

        Returns:
            int: the number of iterations
        """
        if algorithm in ["cc", "kcoreavg", "sssp1"]:
            return 1 # because we predict the sum:  sum * 1 is still the sum. 
        else:
            return self.num_iterations #because we predict the average iteration time. So we need to multiply the prediction.

    def PartitionerSelector(self, data, algorithm):
        data = self.PartitioningQualityPredictor(data=data)    
        data["partitioning_runtime"] = self.PartitioningTimePredictor(data)
        data["processing_runtime"] = self.ProcessingTimePredictor(data, algorithm=algorithm)
        data["end_to_end_runtime"] = data["partitioning_runtime"] + data["processing_runtime"]
        
        partitioner_for_end_to_end_optimization =  data.sort_values(["end_to_end_runtime"])["partitioner"].to_numpy()[0]
        procpartitioner_for_processing_optimizationes =  data.sort_values(["processing_runtime"])["partitioner"].to_numpy()[0]

        print("PartitionerSelector's suggestion: \n Minimize End-to-End Run-time: {} \n Minimize Processing Run-time: {}".format(partitioner_for_end_to_end_optimization, procpartitioner_for_processing_optimizationes))
    
    def PartitioningTimePredictor(self, data):
        return self.partitioning_runtime_model.predict(data) # already in seconds

    def ProcessingTimePredictor(self, data, algorithm):
        return (self.get_num_iterations(algorithm=algorithm) *  self.processing_runtime_models[algorithm].predict(data)) / 1000  # ms to seconds


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

glearner = GraphLearner(
    partitioning_model_name=FINAL_PARTITIONING_MODEL_NAME, 
    partitioning_model_timestamp=FINAL_PARTITIONING_TIMESTAMP,
    processing_model_description=FINAL_PROCESSING_MODELS,
    processing_model_timestamp=FINAL_PROCESSING_TIMESTAMP,
    quality_models_descriptions=FINAL_QUALITY_MODELS_HARD,
    number_iterations=num_iterations) 

def one_hot_encoded_partitioners(graph, selected_partitioners, all_partitioners, num_parts):
    partitioner_dicts = []
    for selected_partitioner in selected_partitioners:
        partitioner_dict = {}
        for partitioner in all_partitioners:
            partitioner_dict[partitioner] = 0
        partitioner_dict[selected_partitioner] = 1
        partitioner_dict["num_partitions"] = num_parts
        partitioner_dict["graph"] = graph
        partitioner_dict["partitioner"] = selected_partitioner
        partitioner_dicts.append(partitioner_dict)
    return partitioner_dicts

input_df = pd.read_csv(metrics_files)
selected_graphs = list(input_df.graph.unique())
all_rows = []
for graph in selected_graphs:
    all_rows += one_hot_encoded_partitioners(graph, selected_partitioners, all_partitioners, num_parts)
input_df = input_df.merge(pd.DataFrame(all_rows), on=["graph"])
glearner.PartitionerSelector(input_df, algorithm=algorithm)