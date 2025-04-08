import pandas as pd
from online_evaluation import online_evaluation
from prediction_algorithm import PredictionAlgorithm
from sklearn.ensemble import  RandomForestRegressor
from sklearn.neighbors import  KNeighborsRegressor
from utils import convert_to_str, regression_report, parse_to_int
import numpy as np 
from xgboost import XGBRegressor
from encodings import int_encoding, sb_encoding 
import os 

if __name__ == "__main__":

    # Define some variable 
    res_path = "results"
    alphas = [15, 30, 45, 60]
    betas = [1,2,5,10]

    # Load the data 
    pm100 = pd.read_parquet("pm100.parquet")
    f_data = pd.read_parquet("f_data.parquet")

    # Create power consumption target for PM100
    pm100["navgpcon"] = pm100.apply(lambda job: np.mean(job.power_consumption/job.num_nodes_alloc), axis = 1).values
    pm100["nminpcon"] = pm100.apply(lambda job: np.min(job.power_consumption/job.num_nodes_alloc), axis = 1).values
    pm100["nmaxpcon"] = pm100.apply(lambda job: np.max(job.power_consumption/job.num_nodes_alloc), axis = 1).values

    # Create power consumption target for F-DATA
    f_data["nmaxpcon"] = f_data.apply(lambda j: int(j.maxpcon/j.nnuma), axis = 1).values
    f_data["navgpcon"] = f_data.apply(lambda j: int(j.avgpcon/j.nnuma), axis = 1).values
    f_data["nminpcon"] = f_data.apply(lambda j: int(j.minpcon/j.nnuma), axis = 1).values

    # Define the input feature set and target feature for PM100
    input_feature_f_data = ["usr", "jnam", "cnumr", "nnumr", "CR-STR-jobenv-req", "CR-STR-freq-req"]
    input_feature_pm100 = ["account", "name", "partition", "qos", "num_cores_req", "mem_req", "num_gpus_req"]

    target_features = ["nmaxpcon", "navgpcon", "nminpcon"]

    for data in ["pm100", "f_data"]:
        if data == "pm100":
            jobs_data = pm100
            input_feature_set = input_feature_pm100
        else:
            jobs_data = f_data
            input_feature_set = input_feature_f_data

        for target in target_features:
            for alpha in alphas:
                experiments_setup = {
                    f"INT+KNN_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": int_encoding, 
                            "classification_model": KNeighborsRegressor(),
                        }
                    },
                    f"INT+RF_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": int_encoding, 
                            "classification_model": RandomForestRegressor(),
                        }
                    },
                    f"INT+XG_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": int_encoding, 
                            "classification_model": XGBRegressor(),
                        }
                    },
                    f"SB+KNN_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": sb_encoding, 
                            "classification_model": KNeighborsRegressor(),
                        }
                    },
                    f"SB+RF_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": sb_encoding, 
                            "classification_model": RandomForestRegressor(),
                        }
                    },
                    f"SB+XG_{target.upper()}": 
                        {
                        "target_feat" : target,
                        "model" : PredictionAlgorithm,
                        "evaluation_function" : regression_report,
                        "feature_mapping_function" : int_encoding, 
                        "hyperparameters" : {
                            "input_feature_set": input_feature_set, 
                            "target_feature": target, 
                            "encoding_foo": sb_encoding, 
                            "classification_model": XGBRegressor(),
                        }
                    },
                }
                online_evaluation(df = jobs_data, experiments_setup_dict=experiments_setup, alpha=alpha, betas=betas, res_path=res_path)



