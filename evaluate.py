import os
import argparse
import json
import sqlite3
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional, Set

def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame for comparison (convert types, handle NULLs consistently)
    
    Args:
        df: Input DataFrame
    
    Returns:
        Normalized DataFrame
    """
    # Convert all columns to string
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Replace 'None' and 'nan' strings with empty string
    df = df.replace(['None', 'nan', 'NULL', 'null'], '')
    
    return df

def compute_soft_f1(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> float:
    """Compute the soft F1 score between predicted and gold DataFrames
    
    Args:
        pred_df: Predicted DataFrame
        gold_df: Gold DataFrame
    
    Returns:
        Soft F1 score
    """
    # Normalize DataFrames for comparison
    pred_df = normalize_table(pred_df)
    gold_df = normalize_table(gold_df)
    
    # Convert DataFrames to sets of tuples for easier comparison
    pred_set = set()
    gold_set = set()
    
    for _, row in pred_df.iterrows():
        pred_set.add(tuple(row.values))
    
    for _, row in gold_df.iterrows():
        gold_set.add(tuple(row.values))
    
    # Calculate metrics for exact matches
    tp = len(pred_set.intersection(gold_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    # Handle special case where both sets are empty
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0
    
    # Handle case where one set is empty
    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def compute_execution_accuracy(
    pred_sql: str, 
    gold_sql: str, 
    db_path: str
) -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Compute execution accuracy by comparing SQL execution results
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold SQL query
        db_path: Path to the SQLite database
    
    Returns:
        Tuple of (is_correct, pred_df, gold_df)
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Execute gold query
        gold_df = pd.read_sql_query(gold_sql, conn)
        
        # Execute predicted query
        pred_df = pd.read_sql_query(pred_sql, conn)
        
        # Close connection
        conn.close()
        
        # Normalize DataFrames
        pred_df = normalize_table(pred_df)
        gold_df = normalize_table(gold_df)
        
        # Check if results match
        is_equal = pred_df.equals(gold_df)
        
        return is_equal, pred_df, gold_df
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return False, None, None

def compute_rves(
    pred_sql: str, 
    gold_sql: str, 
    db_path: str, 
    timeout: float = 3.0,
    num_runs: int = 5
) -> float:
    """Compute Reward-based Valid Efficiency Score (R-VES)
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold SQL query
        db_path: Path to the SQLite database
        timeout: Timeout in seconds
        num_runs: Number of runs for averaging
    
    Returns:
        R-VES score
    """
    # Check execution accuracy first
    is_correct, _, _ = compute_execution_accuracy(pred_sql, gold_sql, db_path)
    
    if not is_correct:
        return 0.0  # Invalid SQL gets 0 score
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Measure execution time for gold SQL
        gold_times = []
        for _ in range(num_runs):
            start_time = time.time()
            cursor.execute(gold_sql)
            cursor.fetchall()
            end_time = time.time()
            gold_times.append(end_time - start_time)
        
        # Average gold execution time
        avg_gold_time = sum(gold_times) / len(gold_times)
        
        # Measure execution time for predicted SQL
        pred_times = []
        for _ in range(num_runs):
            start_time = time.time()
            cursor.execute(pred_sql)
            cursor.fetchall()
            end_time = time.time()
            pred_times.append(end_time - start_time)
        
        # Average predicted execution time
        avg_pred_time = sum(pred_times) / len(pred_times)
        
        # Close connection
        conn.close()
        
        # Calculate time ratio
        time_ratio = avg_pred_time / avg_gold_time if avg_gold_time > 0 else float('inf')
        
        # Calculate R-VES score
        if time_ratio <= 1.0:
            rves = 1.0
        elif time_ratio <= 1.5:
            rves = 0.75
        elif time_ratio <= 2.0:
            rves = 0.5
        elif time_ratio <= 4.0:
            rves = 0.25
        else:
            rves = 0.0
        
        return rves
    except Exception as e:
        print(f"Error calculating R-VES: {e}")
        return 0.0

def evaluate(
    predictions_path: str,
    gold_path: str,
    db_dir: str
) -> Dict[str, float]:
    """Evaluate predictions against gold standard
    
    Args:
        predictions_path: Path to predictions file
        gold_path: Path to gold SQL file
        db_dir: Directory containing the databases
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Load gold SQL queries
    gold_queries = {}
    with open(gold_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t----- bird -----\t')
            if len(parts) == 2:
                sql, db_id = parts
                gold_queries[db_id] = sql
    
    # Initialize metrics
    total = len(predictions)
    correct = 0
    soft_f1_scores = []
    rves_scores = []
    
    # Evaluate each prediction
    for pred in tqdm(predictions, desc="Evaluating"):
        db_id = pred["db_id"]
        pred_sql = pred["sql"]
        
        # Skip if prediction is empty
        if not pred_sql:
            continue
        
        # Get gold SQL
        gold_sql = gold_queries.get(db_id)
        if not gold_sql:
            print(f"Warning: No gold SQL found for {db_id}")
            continue
        
        # Database path
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            continue
        
        # Compute execution accuracy
        is_correct, pred_df, gold_df = compute_execution_accuracy(pred_sql, gold_sql, db_path)
        
        if is_correct:
            correct += 1
        
        # Compute soft F1 score if dataframes are available
        if pred_df is not None and gold_df is not None:
            soft_f1 = compute_soft_f1(pred_df, gold_df)
            soft_f1_scores.append(soft_f1)
        
        # Compute R-VES score
        if is_correct:
            rves = compute_rves(pred_sql, gold_sql, db_path)
            rves_scores.append(rves)
    
    # Calculate overall metrics
    execution_accuracy = correct / total if total > 0 else 0
    avg_soft_f1 = sum(soft_f1_scores) / len(soft_f1_scores) if soft_f1_scores else 0
    avg_rves = sum(rves_scores) / len(rves_scores) if rves_scores else 0
    
    # Combine metrics
    metrics = {
        "execution_accuracy": execution_accuracy * 100,  # as percentage
        "soft_f1": avg_soft_f1 * 100,  # as percentage
        "rves": avg_rves * 100,  # as percentage
        "total_examples": total,
        "correct_examples": correct
    }
    
    return metrics

def evaluate_by_difficulty(
    predictions_path: str,
    gold_path: str,
    db_dir: str,
    difficulty_map_path: str
) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions by difficulty level
    
    Args:
        predictions_path: Path to predictions file
        gold_path: Path to gold SQL file
        db_dir: Directory containing the databases
        difficulty_map_path: Path to difficulty mapping file
    
    Returns:
        Dictionary of metrics by difficulty level
    """
    # Load difficulty mapping
    with open(difficulty_map_path, 'r') as f:
        difficulty_map = json.load(f)
    
    # Group predictions by difficulty
    simple_preds = []
    moderate_preds = []
    challenging_preds = []
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Group by difficulty
    for pred in predictions:
        question_id = f"{pred['db_id']}_{pred['question']}"
        
        if question_id in difficulty_map:
            difficulty = difficulty_map[question_id]
            
            if difficulty == "simple":
                simple_preds.append(pred)
            elif difficulty == "moderate":
                moderate_preds.append(pred)
            elif difficulty == "challenging":
                challenging_preds.append(pred)
    
    # Evaluate each group
    simple_metrics = evaluate(simple_preds, gold_path, db_dir)
    moderate_metrics = evaluate(moderate_preds, gold_path, db_dir)
    challenging_metrics = evaluate(challenging_preds, gold_path, db_dir)
    
    # Combine results
    results = {
        "overall": evaluate(predictions, gold_path, db_dir),
        "simple": simple_metrics,
        "moderate": moderate_metrics,
        "challenging": challenging_metrics
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate BIRD-SQL Mini-Dev predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold SQL file")
    parser.add_argument("--db-dir", type=str, default="./mini_dev_data/dev_databases", help="Directory with databases")
    parser.add_argument("--output", type=str, default="./evaluation_results.json", help="Output path for evaluation results")
    
    args = parser.parse_args()
    
    # Run evaluation
    print(f"Evaluating predictions from {args.predictions}")
    metrics = evaluate(args.predictions, args.gold, args.db_dir)
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Execution Accuracy (EX): {metrics['execution_accuracy']:.2f}%")
    print(f"Soft F1 Score: {metrics['soft_f1']:.2f}%")
    print(f"R-VES: {metrics['rves']:.2f}%")
    print(f"Correct examples: {metrics['correct_examples']} / {metrics['total_examples']}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()