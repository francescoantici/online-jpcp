# online-jpcp

This repository contains the code for the online algorithm for job power prediction in HPC systems, and its evaluation. Below is an overview of the files in this repository:

## File Descriptions

### `encodings.py`
Contains encoding functions for transforming data:
- `sb_encoding`: Converts a DataFrame into a sentence-based representation using the SBert model.
- `int_encoding`: Encodes categorical features into integer codes.

### `utils.py`
Provides utility functions to parse the job data. 

### `t.py`
The main script for running experiments. It:
- Loads datasets (`pm100.parquet` and `f_data.parquet`).
- Prepares input and target features for prediction.
- Defines and runs multiple experiments using different models (e.g., KNN, Random Forest, XGBoost) and encoding methods (integer and sentence-based).
- Calls the `online_evaluation` function to evaluate the models.

### `online_evaluation.py`
Implements the `online_evaluation` function, which:
- Runs online experiments on a dataset using specified configurations.
- Trains the online prediction algorithm on historical data and evaluates them on future data.
- Saves predictions and evaluation metrics to disk.

### `prediction_algorithm.py`
Defines the `PredictionAlgorithm` class, which:
- Prepares data for training and prediction.
- Encodes features using a specified encoding function.
- Fits a classification model and makes predictions.
