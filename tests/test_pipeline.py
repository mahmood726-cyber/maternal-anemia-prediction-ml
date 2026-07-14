import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add code/ path directly to sys.path to avoid name conflict with stdlib 'code'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from train_model import preprocess_nhanes, preprocess_uganda_dhs

def test_preprocess_nhanes_logic():
    # Construct mock NHANES DataFrame
    mock_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4],
        "RIDAGEYR": [25.0, 30.0, 35.0, 40.0],
        "RIDRETH1": [1.0, 3.0, 4.0, 5.0],
        "INDFMPIR": [1.5, np.nan, 3.0, 4.5],  # includes one NaN to test median imputation
        "DMDEDUC2": [4.0, 3.0, 7.0, 9.0],      # includes special values 7 and 9 to test modal imputation
        "DMDMARTL": [1.0, 5.0, 77.0, np.nan],  # includes NaN and 77
        "DMDFMSIZ": [3.0, 4.0, 2.0, 5.0],
        "RIDEXPRG": [1.0, 1.0, 1.0, 1.0],
        "LBXHGB": [10.5, 12.0, np.nan, 11.5]   # includes NaN to test dropna
    })
    
    # Run preprocessor
    X, y = preprocess_nhanes(mock_data)
    
    # Assertions
    # 1. Row 3 is dropped because LBXHGB is NaN, so we expect 3 output rows.
    assert len(X) == 3
    assert len(y) == 3
    
    # 2. Target "anemia" is binary: 10.5 < 11.0 -> 1; 12.0 -> 0; 11.5 -> 0
    np.testing.assert_array_equal(y, [1, 0, 0])
    
    # 3. Median imputation check: INDFMPIR has NaN in index 1. 
    # Valid values: [1.5, 3.0, 4.5]. Median is 3.0.
    # Row with SEQN=2 (idx 1 in output) should have INDFMPIR = 3.0.
    assert X.iloc[1]["INDFMPIR"] == 3.0
    
    # 4. Categorical columns should have been one-hot encoded
    assert any(col.startswith("RIDRETH1_") for col in X.columns)
    assert any(col.startswith("DMDEDUC2_") for col in X.columns)
    assert any(col.startswith("DMDMARTL_") for col in X.columns)

def test_preprocess_uganda_dhs_logic():
    # Construct mock Uganda DHS DataFrame
    # Note: v456 is Hb * 10
    mock_data = pd.DataFrame({
        "v213": [1, 1, 1, 0, 1], # v213=0 should be filtered out
        "v456": [105, 120, np.nan, 115, 999], # nan and 999 should be filtered/dropped
        "v012": [22, 28, 30, 25, 35],
        "v190": [2, 4, 3, 1, 5],
        "v136": [4, 5, 6, 3, 7],
        "v025": [1, 2, 2, 1, 2],
        "v106": [1, 2, 0, 1, 3],
        "v113": [1, 2, 2, 1, 2],
        "v501": [1, 2, 98, 1, 0] # 98 is special code for missing
    })
    
    X, y = preprocess_uganda_dhs(mock_data)
    
    # Expected valid rows:
    # row 0: v213=1, v456=105 (Valid, Anemic)
    # row 1: v213=1, v456=120 (Valid, Non-anemic)
    # row 2: v213=1, v456=NaN (Dropped)
    # row 3: v213=0 (Filtered out by v213==1)
    # row 4: v213=1, v456=999 (Filtered out because Hb > 25g/dL or special code)
    # Therefore, expect 2 rows in output
    assert len(X) == 2
    assert len(y) == 2
    
    # Target "anemia" check: 105 < 110 -> 1; 120 -> 0
    np.testing.assert_array_equal(y, [1, 0])
    
    # Check dummy columns created
    assert any(col.startswith("v025_") for col in X.columns)
    assert any(col.startswith("v106_") for col in X.columns)
    assert any(col.startswith("v113_") for col in X.columns)
    assert any(col.startswith("v501_") for col in X.columns)
