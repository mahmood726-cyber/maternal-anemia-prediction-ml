import os
import urllib.request
import pandas as pd
import numpy as np

# Define six cycles and their verified CDC URLs
CYCLES = {
    "2017-2018": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.xpt"
    },
    "2015-2016": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/CBC_I.xpt"
    },
    "2013-2014": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/CBC_H.xpt"
    },
    "2011-2012": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/CBC_G.xpt"
    },
    "2009-2010": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/CBC_F.xpt"
    },
    "2007-2008": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CBC_E.xpt"
    }
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

def download_file(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_folder, filename)
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Downloaded {filename} successfully.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            raise
    else:
        print(f"Using cached file: {dest_path}")
    return dest_path

def load_and_merge():
    all_pregnant_data = []
    
    for cycle, urls in CYCLES.items():
        print(f"\nProcessing NHANES cycle: {cycle}")
        try:
            demo_path = download_file(urls["demo"], DATA_DIR)
            cbc_path = download_file(urls["cbc"], DATA_DIR)
            
            # Load XPT files using pandas
            df_demo = pd.read_sas(demo_path)
            df_cbc = pd.read_sas(cbc_path)
            
            # Keep necessary variables
            # Demographics: SEQN (ID), RIDAGEYR (Age), RIDRETH1 (Race/Ethnicity), INDFMPIR (Income-to-poverty ratio),
            # DMDEDUC2 (Education level 20+), DMDMARTL (Marital status), DMDFMSIZ (Family size), RIDEXPRG (Pregnancy status)
            demo_cols = ["SEQN", "RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "DMDMARTL", "DMDFMSIZ", "RIDEXPRG"]
            
            demo_cols = [c for c in demo_cols if c in df_demo.columns]
            df_demo_filtered = df_demo[demo_cols].copy()
            
            # Laboratory: SEQN (ID), LBXHGB (Hemoglobin)
            cbc_cols = ["SEQN", "LBXHGB"]
            cbc_cols = [c for c in cbc_cols if c in df_cbc.columns]
            df_cbc_filtered = df_cbc[cbc_cols].copy()
            
            # Merge on SEQN
            merged = pd.merge(df_demo_filtered, df_cbc_filtered, on="SEQN", how="inner")
            
            # Filter for pregnant females: RIDEXPRG == 1 (1 means yes, pregnant)
            pregnant_df = merged[merged["RIDEXPRG"] == 1].copy()
            pregnant_df["cycle"] = cycle
            print(f"Found {len(pregnant_df)} pregnant women in cycle {cycle}.")
            all_pregnant_data.append(pregnant_df)
        except Exception as e:
            print(f"Error processing cycle {cycle}: {e}")
            raise
        
    # Combine all cycles
    combined_df = pd.concat(all_pregnant_data, ignore_index=True)
    print(f"\nCombined dataset size: {len(combined_df)} rows")
    
    # Save raw combined dataset
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_output_path = os.path.join(PROCESSED_DIR, "nhanes_pregnant_raw.csv")
    combined_df.to_csv(raw_output_path, index=False)
    print(f"Saved raw combined dataset to {raw_output_path}")
    
    return combined_df

if __name__ == "__main__":
    load_and_merge()
