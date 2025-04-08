import pandas as pd
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Converts a dictionary-like object (e.g., a Pandas Series) to a comma-separated string.
# Only includes non-null and non-NaN values.
def convert_to_str(job):
    return ",".join([f"{job[k]}" for k in job.index if (job[k] and not(pd.isna(job[k])))])

# Generates a regression evaluation report based on true and predicted values.
# Supports both dictionary and formatted string output.
def regression_report(y_true, y_pred, output_dict:bool = False, digits:int = 2):
    # Define the headers for the report.
    headers = ["mae", "mape", "mse", "rmse", "nrmse", "r2", "support"]
    
    # Calculate regression metrics.
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    mape = mean_absolute_percentage_error(y_true, y_pred)  # Mean Absolute Percentage Error
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    nrmse = rmse / np.mean(y_true)  # Normalized RMSE
    r2 = r2_score(y_true, y_pred)  # R-squared score
    lines = ("score", mae, mape, mse, rmse, nrmse, r2, len(y_true))  # Combine metrics into a tuple.
    
    # If output_dict is True, return the metrics as a dictionary.
    if output_dict:
        return {headers[i]: lines[i + 1] for i in range(len(headers))}
    
    # Format the report as a string.
    width = len("score")  # Determine the width for formatting.
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)  # Header format.
    report = head_fmt.format("", *headers, width=width)  # Generate the header row.
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * len(lines[1:-1]) + " {:>9}\n"  # Row format.
    report += row_fmt.format(*lines, width=width, digits=digits)  # Generate the data row.
    report += "\n"
    return report  # Return the formatted report.

# Safely parses a value to an integer. Returns -1 if the conversion fails.
def parse_to_int(fv):
    try:
        return int(fv)  # Attempt to convert the value to an integer.
    except:  # Catch any exception (e.g., invalid input).
        return -1  # Return -1 if the conversion fails.


