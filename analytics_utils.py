import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, List, Optional

def aggregate_data_by_column(data: List[Dict], group_column: str, value_columns: List[str] = None) -> Dict:
    """
    Aggregate data by a specific column and calculate totals for numeric columns.
    Returns aggregated data suitable for charts and tables.
    """
    if not data:
        return {"data": [], "summary": "No data available"}
    
    # If no value columns specified, use all numeric columns
    if not value_columns:
        sample_row = data[0] if data else {}
        value_columns = [col for col, val in sample_row.items() 
                        if isinstance(val, (int, float)) and col != group_column]
    
    # Group data by the specified column
    grouped_data = {}
    for row in data:
        group_value = row.get(group_column, "Unknown")
        if group_value not in grouped_data:
            grouped_data[group_value] = []
        grouped_data[group_value].append(row)
    
    # Calculate totals for each group
    aggregated_data = []
    for group_value, group_rows in grouped_data.items():
        group_summary = {group_column: group_value}
        
        # Calculate totals for each value column
        for col in value_columns:
            total = sum(row.get(col, 0) for row in group_rows 
                       if isinstance(row.get(col), (int, float)))
            group_summary[f"Total_{col}"] = total
            group_summary[f"Count_{col}"] = len([row for row in group_rows 
                                               if row.get(col) is not None])
        
        aggregated_data.append(group_summary)
    
    # Sort by the group column for better presentation
    aggregated_data.sort(key=lambda x: str(x.get(group_column, "")))
    
    return {
        "data": aggregated_data,
        "summary": f"Aggregated by {group_column}",
        "total_groups": len(aggregated_data),
        "value_columns": value_columns
    }

def clean_and_validate_data(data: List[Dict]) -> List[Dict]:
    """
    Clean and validate data before aggregation.
    Removes None values, converts types, and handles edge cases.
    """
    if not data:
        return []
    
    cleaned_data = []
    for row in data:
        if not isinstance(row, dict):
            continue
        
        cleaned_row = {}
        for key, value in row.items():
            # Handle None, empty strings, and whitespace
            if value is None or (isinstance(value, str) and value.strip() == ""):
                cleaned_row[key] = "Unknown"
            elif isinstance(value, str):
                cleaned_row[key] = value.strip()
            else:
                cleaned_row[key] = value
        
        cleaned_data.append(cleaned_row)
    
    return cleaned_data

def count_by_column(data: List[Dict], count_column: str) -> Dict:
    """
    Count occurrences of each unique value in a specific column.
    Useful for requests like "count by MHU" or "patients by location".
    """
    if not data:
        return {"data": [], "summary": "No data available"}
    
    # Clean the data first
    cleaned_data = clean_and_validate_data(data)
    
    if not cleaned_data:
        return {"data": [], "summary": "No valid data after cleaning"}
    
    # Log the first few rows for debugging
    logger = logging.getLogger(__name__)
    logger.info(f"Counting by column: {count_column}")
    logger.info(f"Sample data (first 3 rows): {cleaned_data[:3]}")
    logger.info(f"Available columns: {list(cleaned_data[0].keys()) if cleaned_data else []}")
    
    # Check if the column exists
    if count_column not in cleaned_data[0]:
        logger.warning(f"Column '{count_column}' not found in data. Available columns: {list(cleaned_data[0].keys())}")
        return {"data": [], "summary": f"Column '{count_column}' not found in data"}
    
    # Count occurrences
    counts = {}
    total_rows = 0
    
    for row in cleaned_data:
        total_rows += 1
        value = row.get(count_column, "Unknown")
        
        # Handle different data types
        if value is None:
            value = "Unknown"
        elif isinstance(value, (int, float)):
            value = str(value)  # Convert numbers to strings for consistent counting
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                value = "Unknown"
        
        counts[value] = counts.get(value, 0) + 1
    
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Unique values found: {len(counts)}")
    logger.info(f"Count breakdown: {counts}")
    
    # Convert to list format for charts/tables
    count_data = [{"Category": key, "Count": value} for key, value in counts.items()]
    
    # Sort by count (descending) for better presentation
    count_data.sort(key=lambda x: x["Count"], reverse=True)
    
    return {
        "data": count_data,
        "summary": f"Count by {count_column}",
        "total_categories": len(count_data),
        "total_count": sum(counts.values()),
        "debug_info": {
            "total_rows_processed": total_rows,
            "column_found": count_column in cleaned_data[0] if cleaned_data else False,
            "available_columns": list(cleaned_data[0].keys()) if cleaned_data else []
        }
    }

def get_numeric_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    """Return mean, median, std, min, max for all numeric columns."""
    summaries = {}
    for col in df.select_dtypes(include='number').columns:
        arr = df[col].dropna().values
        if arr.size > 0:
            summaries[col] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'sum': float(np.sum(arr)),
            }
    return summaries

def predict_trend(df: pd.DataFrame, x_col: str, y_col: str, periods_ahead: int = 1) -> Optional[float]:
    """
    Fit a linear regression to predict y_col from x_col and forecast periods_ahead into the future.
    x_col should be numeric or datetime (will be converted to ordinal).
    Returns the predicted y value for the next period.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = df[x_col]
    y = df[y_col]
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.map(pd.Timestamp.toordinal)
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)
    if len(X) < 2:
        return None
    model = LinearRegression()
    model.fit(X, y)
    next_x = X[-1][0] + periods_ahead
    y_pred = model.predict(np.array([[next_x]]))
    return float(y_pred[0]) 