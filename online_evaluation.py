from datetime import datetime, timedelta
import time
import numpy as np
from tqdm import tqdm
import os
import pickle

def online_evaluation(df, experiments_setup_dict = {}, alpha = 15, betas = [1], res_path = "results", data_params = [], st = None, et = None):    
    """
    
    Run online experiments on the given dataframe using the specified configurations.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    experiments_setup_dict (dict): A dictionary containing the experiment configurations.
    alpha (int): The size of the training window.
    betas (list): A list of beta values representing the size of the evaluation window.
    res_path (str): The path to store the results.
    data_params (list): A list of additional parameters to save alongside  the results.
    st (datetime): The start date for evaluation.
    et (datetime): The end date for evaluation."""   

    # Create directory paths for storing prediction results
    dict_path = os.path.join(res_path, "pred_dict")
    for p in [res_path, dict_path]:
        if not(os.path.exists(p)):
            os.mkdir(p)
    
    # Convert UNIX timestamps to date objects
    df["day"] = df.adt.apply(lambda adt: datetime.fromtimestamp(int(adt)).date())
    df["end_day"] = df.edt.apply(lambda edt: datetime.fromtimestamp(int(edt)).date())
        
    # Get unique days in the dataset
    days = df.day.unique()
    
    if alpha:
        min_alpha = alpha

    # Define the starting day of evaluation
    st_value = st if st else days.min() + timedelta(days=min_alpha)
    
    # Find index of start day in the array of days
    st_idx = np.where(days == st_value)[0][0]

    # Map each eligible evaluation day to its corresponding betas (evaluation window sizes)
    beta_day_dict = {
        day: [beta for beta in betas if day in list(map(lambda d: days[d], range(st_idx, len(days), beta)))]
        for day in days[st_idx:]
    }

    # Initialize dictionary to store evaluation outputs
    betas_out_dict = {}
    for beta in betas:
        res_dict_path = os.path.join(dict_path, f"pred_{alpha}_{beta}.pickle") 
        if os.path.exists(res_dict_path):
            with open(res_dict_path, "rb") as f:
                betas_out_dict[beta] = pickle.load(f)
        else:
            betas_out_dict[beta] = {}
    
    # Iterate through each evaluation day
    for edge_day in tqdm(beta_day_dict):
        betas_eligible = beta_day_dict[edge_day]
        if len(beta_day_dict) == 0:
            continue

        # Select historical training data before current evaluation day (with window size alpha)
        if alpha:
            train_df = df[(df.end_day >= edge_day - timedelta(days=alpha)) & (df.end_day < edge_day)]
        else:
            train_df = df[df.end_day < edge_day]
                            
        if len(train_df) == 0:
            continue

        # Run all experiment configurations
        for experiment_name in experiments_setup_dict:
            experiment_setting = experiments_setup_dict[experiment_name]
            
            target_feat = experiment_setting["target_feat"]
            model = experiment_setting["model"]
                                    
            # Train the model
            try:
                t0_t = time.time()
                model = model(**experiment_setting["hyperparameters"]).fit(train_df)
                t1_t = time.time()
            except:
                continue
            
            # For each eligible beta, evaluate model on corresponding test window
            for beta in betas_eligible:
                betas_out_dict[beta][experiment_name] = betas_out_dict[beta].get(experiment_name, {})
                betas_out_dict[beta][experiment_name][str(edge_day)] = betas_out_dict[beta][experiment_name].get(
                    str(edge_day), {p:[] for p in data_params + ["true", "pred", "time_train", "time_inf"]})
                      
                try:
                    # Select test data between [edge_day, edge_day + beta)
                    test_df = df[(df.day >= edge_day) & (df.day < edge_day + timedelta(days = beta))].sort_values("adt")
                    
                    # Predict using trained model
                    t0_i = time.time()
                    y_pred = list(model.predict(test_df)) 
                    t1_i = time.time()
                except:
                    continue
                else: 
                    # Store true values, predictions, any additional params, and timings
                    betas_out_dict[beta][experiment_name][str(edge_day)]["true"] = list(test_df[target_feat].values)
                    betas_out_dict[beta][experiment_name][str(edge_day)]["pred"] = y_pred
                    for p in data_params:
                        betas_out_dict[beta][experiment_name][str(edge_day)][p] = list(test_df[p].values)
                    betas_out_dict[beta][experiment_name][str(edge_day)]["time_train"].append(t1_t-t0_t)
                    betas_out_dict[beta][experiment_name][str(edge_day)]["time_inf"].append(t1_i-t0_i)

    # Save results for each beta to disk
    for beta in betas:  
        res_dict_path = os.path.join(dict_path, f"pred_{alpha}_{beta}pickle")                             
        with open(res_dict_path, "wb") as res_d:
            pickle.dump(betas_out_dict[beta], res_d)   

    # Evaluate and save performance metrics
    for experiment_name in experiments_setup_dict:
        try:            
            true = []
            preds = []
            for beta in betas:
                for day in betas_out_dict[beta][experiment_name]:
                    true += betas_out_dict[beta][experiment_name][day]["true"]
                    preds += betas_out_dict[beta][experiment_name][day]["pred"]
                
                # Call evaluation function and save result to text file
                with open(os.path.join(res_path, f"results_{alpha}_{beta}_{experiment_name}.txt"), "w") as f:
                    f.write(experiments_setup_dict[experiment_name]["evaluation_function"](true, preds))
                    
        except Exception as e:
            print(e)
            continue
