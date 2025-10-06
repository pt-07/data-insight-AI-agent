import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json

def profile_dataset(df, target_column=None, dataset_name="dataset"):
    """
    Comprehensive feature profiling for a dataset
    Returns analysis suitable for AI agent consumption
    """
    profile = {
        "dataset_name": dataset_name,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": {},
        "summary_stats": {},
        "correlations": {},
        "feature_importance": {},
        "data_quality": {},
        "sample_data": {}
    }
    #check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    profile["missing_values"] = {
        col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        profile["summary_stats"][col] = {
            "type": "numeric",
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "unique_count": int(df[col].nunique())
        }
    
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10)
        profile["summary_stats"][col] = {
            "type": "categorical",
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
            "most_common": str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None
        }
    
    #numeric correlations
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        #top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })
        #absolute correlation
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        profile["correlations"] = corr_pairs[:15]  # Top 15
    
    #how important each feature is
    if target_column and target_column in df.columns:
        importance = calculate_feature_importance(df, target_column)
        profile["feature_importance"] = importance
    profile["data_quality"] = assess_data_quality(df)
    profile["sample_data"] = df.head(5).to_dict('records')
    return profile

def calculate_feature_importance(df, target_column, max_features=15):
    """Calculate feature importance using Random Forest"""
    try:
        #clean and prep data
        df_clean = df.dropna(subset=[target_column])
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Encode categorical variables
        le_dict = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
        
        # Determine if classification or regression
        if y.dtype == 'object' or y.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importance
        importance_scores = dict(zip(X.columns, model.feature_importances_))
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "target": target_column,
            "scores": {k: float(v) for k, v in sorted_importance[:max_features]}
        }
    except Exception as e:
        return {"error": str(e)}

def assess_data_quality(df):
    """Assess data quality issues"""
    issues = []
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        issues.append({
            "type": "high_missing_values",
            "columns": list(high_missing.index),
            "severity": "high"
        })
    #duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "count": int(dup_count),
            "percentage": float(dup_count / len(df) * 100),
            "severity": "medium"
        })
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            issues.append({
                "type": "zero_variance",
                "column": col,
                "severity": "low"
            })
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            issues.append({
                "type": "high_cardinality",
                "column": col,
                "unique_ratio": float(unique_ratio),
                "severity": "medium"
            })
    
    return issues

def profile_multiple_datasets(datasets_dict, relationships=None):
    """
    Profile multiple related datasets (e.g., orders, products, customers)
    datasets_dict: {name: dataframe}
    relationships: list of dicts with 'dataset1', 'dataset2', 'key' fields
    """
    profiles = {}
    for name, df in datasets_dict.items():
        profiles[name] = profile_dataset(df, dataset_name=name)
    if relationships:
        profiles["relationships"] = relationships
    profiles["overview"] = {
        "total_datasets": len(datasets_dict),
        "total_rows": sum(len(df) for df in datasets_dict.values()),
        "total_columns": sum(len(df.columns) for df in datasets_dict.values())
    }
    return profiles

# Example usage
if __name__ == "__main__":
    #sample dataframe
    sample_df = pd.DataFrame({
        'order_id': range(1000),
        'product_id': np.random.randint(1, 50, 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'price': np.random.uniform(5, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
    })
    
    profile = profile_dataset(sample_df, target_column='category')
    print(json.dumps(profile, indent=2))