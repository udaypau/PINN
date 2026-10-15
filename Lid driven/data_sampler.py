# The objective of this library is to function as a dataset accumulator, processor, and sampler
# for PINN training and validation workflows.

import os
import glob
import pandas as pd
import numpy as np


def data_aggregator(
    folder_path=None, 
    sample_ratio=0.1, 
    mode='r', 
    inspection=False, 
    random_state=42
):
    """
    Unified dataset accumulator, inspector, and sampler.
    
    Parameters:
      folder_path   (str) : Directory containing the target .csv files.
      sample_ratio  (float): Fraction of points to sample (0.0 < sample_ratio <= 1.0).
      mode          (str) : Sampling mode:
                            'h' -> Display instruction list and exit.
                            'r' -> Random uniform sampling.
                            's' -> Systematic stride sampling (every k-th row).
                            'f' -> First-N fraction (head).
                            'l' -> Last-N fraction (tail).
      inspection    (bool): If True, inspects files, prints dimensions & columns,
                            and exits without concatenating or sampling.
      random_state  (int) : Seed for reproducible random sampling (mode 'r').
      
    Returns:
      dict / pd.DataFrame / None:
        - If mode == 'h'        : None
        - If inspection == True : Summary dictionary {filename: {dimensions, columns}}
        - Default               : Sampled pd.DataFrame
    """
    # 1. Help instruction handler
    if mode.strip().lower() == 'h':
        help_text = """
================================================================================
                    DATA AGGREGATOR & SAMPLER: HELP MENU
================================================================================
Usage:
  data_aggregator(folder_path, sample_ratio=0.1, mode='r', inspection=False, random_state=42)

Parameters:
  folder_path   : Path to folder containing target .csv simulation files.
  sample_ratio  : Float between (0.0, 1.0] specifying sampling fraction (e.g. 0.15 = 15%).
  inspection    : Boolean flag. If True, inspects file metadata and returns a summary dict.
  random_state  : Integer seed for reproducible random selection.
  mode          : Single-character operation flag:
                  'h' - Display this instruction menu and return None.
                  'r' - Random uniform sampling (shuffles rows, preserves distributions).
                  's' - Systematic stride sampling (step = 1/ratio, maintains spatial order).
                  'f' - Head slice (extracts first sample_ratio fraction of points).
                  'l' - Tail slice (extracts last sample_ratio fraction of points).
================================================================================
"""
        print(help_text)
        return None

    # 2. Directory validation
    if folder_path is None or not os.path.isdir(folder_path):
        raise ValueError(f"Valid 'folder_path' directory must be provided. Got: {folder_path}")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {folder_path}")

    # 3. Inspection-Only Mode
    if inspection:
        summary = {}
        print(f"Found {len(csv_files)} CSV file(s) in '{folder_path}':\n" + "-" * 60)
        for filepath in sorted(csv_files):
            filename = os.path.basename(filepath)
            try:
                df = pd.read_csv(filepath)
                cols = df.columns.str.strip().tolist()
                shape = df.shape
                summary[filename] = {"dimensions": shape, "columns": cols}
                print(f"File: {filename}")
                print(f"  Dimensions: {shape[0]:,} rows × {shape[1]} columns")
                print(f"  Columns   : {cols}")
                print("-" * 60)
            except Exception as e:
                print(f"File: {filename} -> Error reading file: {e}")
                print("-" * 60)
        return summary

    # 4. Aggregation and Sampling Mode
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be between 0.0 and 1.0. Got: {sample_ratio}")

    dataframes = []
    print(f"Discovered {len(csv_files)} CSV files. Reading and accumulating...")

    for filepath in sorted(csv_files):
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            dataframes.append(df)
        except Exception as e:
            print(f"Skipping {os.path.basename(filepath)} due to read error: {e}")

    if not dataframes:
        raise ValueError("No valid CSV data could be loaded.")

    master_df = pd.concat(dataframes, axis=0, ignore_index=True)
    total_points = len(master_df)
    n_samples = max(1, int(total_points * sample_ratio))
    print(f"\nConcatenation Complete.")
    print(f"Master Dataset Shape: {master_df.shape} ({total_points:,} total points)")

    # 5. Execute Sampling Strategy
    mode_char = mode.strip().lower()

    if mode_char == 'r':
        sampled_df = master_df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        sample_desc = "Random Uniform"

    elif mode_char == 's':
        step = max(1, int(1.0 / sample_ratio))
        sampled_df = master_df.iloc[::step].iloc[:n_samples].reset_index(drop=True)
        sample_desc = f"Systematic (stride step={step})"

    elif mode_char == 'f':
        sampled_df = master_df.iloc[:n_samples].reset_index(drop=True)
        sample_desc = "First Fraction (Head)"

    elif mode_char == 'l':
        sampled_df = master_df.iloc[-n_samples:].reset_index(drop=True)
        sample_desc = "Last Fraction (Tail)"

    else:
        raise ValueError(f"Unrecognized mode '{mode}'. Pass mode='h' to see valid options.")

    print(f"Mode applied: {sample_desc}")
    print(f"Sampled Dataset Shape: {sampled_df.shape} ({len(sampled_df):,} points at {sample_ratio:.1%})\n")

    return sampled_df