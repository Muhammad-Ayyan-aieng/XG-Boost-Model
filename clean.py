# 01_clean.py
"""
Clean US Accidents dataset (7.7M rows)
Output: datasets/cleaned_data.csv
"""

import pandas as pd
import numpy as np
import os
import re

# Configuration
DATA_PATH = "datasets/US_Accidents_March23.csv"
OUTPUT_PATH = "datasets/cleaned_data.csv"
CHUNK_SIZE = 50000
MIN_KEYWORD_FREQ = 100

def get_time_category(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 20:
        return 'Evening'
    elif 20 <= hour < 24:
        return 'Night'
    return 'Late Night'

def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    return 'Winter'

def extract_keywords(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    keywords = []
    
    patterns = {
        'jackknife': r'jackknife',
        'blocked': r'blocked',
        'multi_vehicle': r'multi-vehicle|multi vehicle',
        'rollover': r'rollover',
        'road_closed': r'road closed',
        'slow_traffic': r'slow traffic',
        'queueing': r'queueing',
        'shoulder': r'shoulder',
        'fatality': r'fatality|fatal',
        'injury': r'injury|injured',
        'ice': r'black ice|ice|icy',
        'snow': r'snow|snowing',
        'fog': r'fog|foggy',
        'heavy_rain': r'heavy rain|downpour'
    }
    
    for kw, pat in patterns.items():
        if re.search(pat, text):
            keywords.append(kw)
    
    return keywords

def process_chunk(chunk):
    # Remove invalid severity
    if 'Severity' in chunk.columns:
        chunk['Severity'] = pd.to_numeric(chunk['Severity'], errors='coerce')
        chunk = chunk[chunk['Severity'].between(1, 4, inclusive='both')]
    
    if len(chunk) == 0:
        return chunk
    
    # Remove rows with >50% missing values
    min_thresh = int(len(chunk.columns) * 0.5)
    chunk = chunk.dropna(thresh=min_thresh)
    
    # Fill missing numerical with median
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    for col in num_cols:
        if col in chunk.columns and chunk[col].isnull().any():
            chunk[col] = chunk[col].fillna(chunk[col].median())
    
    # Fill missing categorical with mode
    cat_cols = ['Weather_Condition', 'Wind_Direction', 'Sunrise_Sunset']
    for col in cat_cols:
        if col in chunk.columns and chunk[col].isnull().any():
            mode_val = chunk[col].mode()
            if len(mode_val) > 0:
                chunk[col] = chunk[col].fillna(mode_val[0])
            else:
                chunk[col] = chunk[col].fillna('UNKNOWN')
    
    # Remove outliers
    if 'Temperature(F)' in chunk.columns:
        chunk = chunk[(chunk['Temperature(F)'] >= -50) & (chunk['Temperature(F)'] <= 130)]
    if 'Humidity(%)' in chunk.columns:
        chunk = chunk[(chunk['Humidity(%)'] >= 0) & (chunk['Humidity(%)'] <= 100)]
    if 'Visibility(mi)' in chunk.columns:
        chunk = chunk[chunk['Visibility(mi)'] >= 0]
    
    # Parse dates
    if 'Start_Time' in chunk.columns:
        chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'], errors='coerce')
        chunk = chunk[chunk['Start_Time'].notna()]
        chunk['Hour'] = chunk['Start_Time'].dt.hour
        chunk['Month'] = chunk['Start_Time'].dt.month
        chunk['DayOfWeek'] = chunk['Start_Time'].dt.dayofweek
        chunk['IsWeekend'] = chunk['DayOfWeek'].isin([5, 6]).astype(int)
        chunk['TimeOfDay'] = chunk['Hour'].apply(get_time_category)
        chunk['Season'] = chunk['Month'].apply(get_season)
    
    if 'End_Time' in chunk.columns and 'Start_Time' in chunk.columns:
        chunk['End_Time'] = pd.to_datetime(chunk['End_Time'], errors='coerce')
        chunk['Duration_Minutes'] = (chunk['End_Time'] - chunk['Start_Time']).dt.total_seconds() / 60
        chunk['Duration_Minutes'] = chunk['Duration_Minutes'].fillna(0).clip(0, 1440)
    
    # Extract keywords
    if 'Description' in chunk.columns:
        kw_cols = ['has_jackknife', 'has_blocked', 'has_multi_vehicle', 
                   'has_rollover', 'has_road_closed', 'has_slow_traffic',
                   'has_queueing', 'has_shoulder', 'has_fatality', 'has_injury',
                   'has_ice', 'has_snow', 'has_fog', 'has_heavy_rain']
        for col in kw_cols:
            chunk[col] = 0
        
        for idx, desc in chunk['Description'].items():
            kw_list = extract_keywords(desc)
            for kw in kw_list:
                col_name = f'has_{kw}'
                if col_name in chunk.columns:
                    chunk.loc[idx, col_name] = 1
    
    # Combination features
    if 'Weather_Condition' in chunk.columns and 'Junction' in chunk.columns:
        chunk['Rain_Junction'] = ((chunk['Weather_Condition'].str.contains('RAIN', na=False)) & 
                                   (chunk['Junction'] == True)).astype(int)
    
    if 'Hour' in chunk.columns:
        chunk['Rush_Hour'] = ((chunk['Hour'].between(7, 9)) | (chunk['Hour'].between(16, 18))).astype(int)
        chunk['Late_Night'] = (chunk['Hour'].between(0, 5)).astype(int)
    
    # Keep only useful columns
    useful_cols = [
        'Severity', 'State', 'City', 'County',
        'Temperature(F)', 'Humidity(%)', 'Precipitation(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition',
        'Hour', 'Month', 'DayOfWeek', 'IsWeekend', 'TimeOfDay', 'Season',
        'Sunrise_Sunset', 'Duration_Minutes',
        'Junction', 'Traffic_Signal', 'Crossing', 'Railway', 'Stop',
        'Traffic_Calming', 'Roundabout', 'Amenity', 'Station', 'No_Exit',
        'Rain_Junction', 'Rush_Hour', 'Late_Night',
        'has_jackknife', 'has_blocked', 'has_multi_vehicle', 'has_rollover',
        'has_road_closed', 'has_slow_traffic', 'has_queueing', 'has_shoulder',
        'has_fatality', 'has_injury', 'has_ice', 'has_snow', 'has_fog', 'has_heavy_rain'
    ]
    existing = [c for c in useful_cols if c in chunk.columns]
    chunk = chunk[existing]
    
    return chunk

def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output exists: {OUTPUT_PATH}")
        print("Delete file to re-run")
        return
    
    print(f"Reading: {DATA_PATH}")
    print(f"Writing: {OUTPUT_PATH}")
    
    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, 
                         on_bad_lines='skip', low_memory=False)
    
    first = True
    total_read = 0
    total_valid = 0
    
    for i, chunk in enumerate(reader, 1):
        original = len(chunk)
        total_read += original
        
        chunk = process_chunk(chunk)
        valid = len(chunk)
        total_valid += valid
        
        if valid == 0:
            continue
        
        if first:
            chunk.to_csv(OUTPUT_PATH, index=False)
            first = False
        else:
            chunk.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        
        if i % 10 == 0:
            print(f"Chunk {i}: read {total_read:,}, valid {total_valid:,}")
    
    print(f"Complete: {total_valid:,} valid rows out of {total_read:,}")
    
    # Filter rare keywords
    print("Filtering rare keywords...")
    df = pd.read_csv(OUTPUT_PATH)
    kw_cols = [c for c in df.columns if c.startswith('has_')]
    removed = 0
    for col in kw_cols:
        if df[col].sum() < MIN_KEYWORD_FREQ:
            df = df.drop(columns=[col])
            removed += 1
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Removed {removed} rare keyword columns")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    main()