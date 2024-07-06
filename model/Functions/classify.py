import pandas as pd

def Classify(data,target):
  # Check if target variable exists
  if 'target' not in data.columns:
    raise ValueError("Dataset must have a 'target' column")

  target = data[target]

  # Check for at least two unique values (0 and 1 possible for classification)
  if target.nunique() < 2:
    print("1")
    return False

  # Check for numeric data type, allowing for 0 and 1
  if pd.api.types.is_numeric_dtype(target):
    # Check if only 0 and 1 are present (strong indicator of classification)
    unique_values = target.unique()
    if len(set(unique_values)) <10:
      print("2")
      return True

  # Less certain case, could be regression or classification with more categories
  print("3")
  return False



