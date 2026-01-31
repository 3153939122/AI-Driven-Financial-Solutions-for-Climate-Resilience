"""
climate_vulnerability_analysis.py
Complete data processing and analysis pipeline
Author: [Your Name]
Date: [Date]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CREATE RAW DATASET
# ============================================

def create_raw_dataset():
    """Create the raw dataset as provided in the manuscript"""
    
    data = {
        'Country': ['Bangladesh', 'Colombia', 'Kenya', 'Morocco', 'Pakistan', 'Vietnam'],
        'Urban_Pop_2024': [41.23, 82.65, 30.05, 65.64, 38.37, 40.20],
        'Agri_Land_2000': [72.21, 40.43, 46.86, 68.68, 47.61, 28.23],
        'Poverty_365_Year': [30.0, 12.2, 70.1, 9.8, 39.8, 3.8],
        'Gini_Index_Year': [33.4, 53.9, 38.4, 39.5, 29.6, 36.8],
        'Historical_Temp_Trend': [0.16, 0.22, 0.21, 0.43, 0.47, np.nan],
        'Projected_Temp_Change_2050': [1.75, 1.50, 1.70, 1.75, 2.50, 1.50],
        'Projected_Precip_Change_2050': ['Slight Increase', 'Slight Decrease', 'Slight Increase', 
                                        'Decrease (-14%)', 'Highly Uncertain', 'N/A'],
        'Drought_Risk_SPEI': ['High', 'Medium', 'Very High', 'Very High', 'Extreme', 'N/A'],
        'Sea_Level_Rise_2050': [0.30, 0.265, np.nan, 0.18, 0.44, 0.44]
    }
    
    df = pd.DataFrame(data)
    return df

# ============================================
# 2. DATA PREPROCESSING FUNCTIONS
# ============================================

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    
    df_clean = df.copy()
    
    # Impute missing temperature trend (Vietnam) with mean
    if df_clean['Historical_Temp_Trend'].isna().any():
        mean_trend = df_clean['Historical_Temp_Trend'].mean()
        df_clean['Historical_Temp_Trend'].fillna(mean_trend, inplace=True)
        print(f"Imputed missing Historical_Temp_Trend with mean: {mean_trend:.3f}")
    
    # Impute missing Sea Level Rise (Kenya) with median
    if df_clean['Sea_Level_Rise_2050'].isna().any():
        median_slr = df_clean['Sea_Level_Rise_2050'].median()
        df_clean['Sea_Level_Rise_2050'].fillna(median_slr, inplace=True)
        print(f"Imputed missing Sea_Level_Rise_2050 with median: {median_slr:.3f}")
    
    return df_clean

def encode_qualitative_variables(df):
    """Encode qualitative variables to numeric values"""
    
    df_encoded = df.copy()
    
    # Drought Risk Encoding (1-5 scale)
    drought_mapping = {
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Very High': 4,
        'Extreme': 5,
        'N/A': np.nan
    }
    
    # Precipitation Change Encoding (-1 to 1 scale)
    precip_mapping = {
        'Decrease (-14%)': -1.0,
        'Slight Decrease': -0.5,
        'Stable': 0.0,
        'Slight Increase': 0.5,
        'Increase': 1.0,
        'Highly Uncertain': 0.0,
        'N/A': np.nan
    }
    
    df_encoded['Drought_Risk_encoded'] = df_encoded['Drought_Risk_SPEI'].map(drought_mapping)
    df_encoded['Precip_Change_encoded'] = df_encoded['Projected_Precip_Change_2050'].map(precip_mapping)
    
    # Impute missing encoded values
    df_encoded['Drought_Risk_encoded'].fillna(df_encoded['Drought_Risk_encoded'].median(), inplace=True)
    df_encoded['Precip_Change_encoded'].fillna(df_encoded['Precip_Change_encoded'].median(), inplace=True)
    
    return df_encoded

def normalize_data(df, numeric_columns):
    """Apply z-score normalization to numeric columns"""
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    
    # Normalize only the specified columns
    df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Save the scaler for future use
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    
    return df_normalized, scaler

# ============================================
# 3. CLUSTERING ANALYSIS (K-Means & DBSCAN)
# ============================================

def perform_clustering(df_normalized, n_clusters=3):
    """Perform K-Means clustering analysis"""
    
    # Features for clustering (all numeric features)
    features = [
        'Urban_Pop_2024', 'Agri_Land_2000', 'Poverty_365_Year',
        'Gini_Index_Year', 'Historical_Temp_Trend',
        'Projected_Temp_Change_2050', 'Drought_Risk_encoded',
        'Precip_Change_encoded', 'Sea_Level_Rise_2050'
    ]
    
    X = df_normalized[features]
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_normalized['KMeans_Cluster'] = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, df_normalized['KMeans_Cluster'])
    print(f"K-Means Silhouette Score: {silhouette_avg:.3f}")
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    df_normalized['DBSCAN_Cluster'] = dbscan.fit_predict(X)
    
    # Count clusters and noise points
    unique_clusters = np.unique(df_normalized['DBSCAN_Cluster'])
    n_clusters_dbscan = len(unique_clusters[unique_clusters != -1])
    n_noise = list(df_normalized['DBSCAN_Cluster']).count(-1)
    
    print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")
    
    return df_normalized, kmeans, dbscan

# ============================================
# 4. COMPOUND VULNERABILITY SCORE (CVS)
# ============================================

def calculate_cvs(df):
    """Calculate Compound Vulnerability Score"""
    
    df_cvs = df.copy()
    
    # Calculate Exposure Score (E)
    exposure_features = ['Projected_Temp_Change_2050', 'Drought_Risk_encoded', 'Sea_Level_Rise_2050']
    df_cvs['Exposure_Score'] = df_cvs[exposure_features].mean(axis=1)
    
    # Calculate Sensitivity Score (S)
    sensitivity_features = ['Poverty_365_Year', 'Gini_Index_Year', 'Agri_Land_2000']
    df_cvs['Sensitivity_Score'] = df_cvs[sensitivity_features].mean(axis=1)
    
    # Get Trend Score (T) - already normalized
    df_cvs['Trend_Score'] = df_cvs['Historical_Temp_Trend']
    
    # Calculate CVS: (E × S) + T
    df_cvs['CVS_Raw'] = (df_cvs['Exposure_Score'] * df_cvs['Sensitivity_Score']) + df_cvs['Trend_Score']
    
    # Normalize CVS to 0-100 scale
    cvs_min = df_cvs['CVS_Raw'].min()
    cvs_max = df_cvs['CVS_Raw'].max()
    df_cvs['CVS_Normalized'] = 100 * (df_cvs['CVS_Raw'] - cvs_min) / (cvs_max - cvs_min)
    
    return df_cvs

# ============================================
# 5. XGBOOST PREDICTION MODEL
# ============================================

def train_xgboost_model(df, target_col='CVS_Normalized'):
    """Train XGBoost model for vulnerability prediction"""
    
    # Features for prediction
    features = [
        'Urban_Pop_2024', 'Poverty_365_Year', 'Historical_Temp_Trend',
        'Projected_Temp_Change_2050', 'Drought_Risk_encoded'
    ]
    
    X = df[features]
    y = df[target_col]
    
    # Initialize XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # Since small dataset, use cross-validation
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    print(f"XGBoost Cross-Validation R² Scores: {cv_scores}")
    print(f"Average R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Train final model
    model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    return model, feature_importance

def predict_future_scenario(df, model, poverty_increase=0.15):
    """Predict future vulnerability under stress scenario"""
    
    df_future = df.copy()
    
    # Apply stress scenario: increase poverty for vulnerable countries
    high_vuln_countries = ['Kenya', 'Pakistan', 'Bangladesh']
    mask = df_future['Country'].isin(high_vuln_countries)
    df_future.loc[mask, 'Poverty_365_Year'] = df_future.loc[mask, 'Poverty_365_Year'] * (1 + poverty_increase)
    
    # Prepare features for prediction
    features = [
        'Urban_Pop_2024', 'Poverty_365_Year', 'Historical_Temp_Trend',
        'Projected_Temp_Change_2050', 'Drought_Risk_encoded'
    ]
    
    X_future = df_future[features]
    
    # Predict future CVS
    df_future['CVS_Future_Predicted'] = model.predict(X_future)
    
    # Calculate change
    df_future['CVS_Change'] = df_future['CVS_Future_Predicted'] - df_future['CVS_Normalized']
    
    return df_future

# ============================================
# 6. MAIN ANALYSIS PIPELINE
# ============================================

def main():
    """Main analysis pipeline"""
    
    print("=" * 60)
    print("CLIMATE VULNERABILITY ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Create raw dataset
    print("\n1. Creating raw dataset...")
    df_raw = create_raw_dataset()
    print(f"   Countries: {', '.join(df_raw['Country'].tolist())}")
    print(f"   Total indicators: {len(df_raw.columns) - 1}")
    
    # Step 2: Handle missing values
    print("\n2. Handling missing values...")
    df_clean = handle_missing_values(df_raw)
    
    # Step 3: Encode qualitative variables
    print("\n3. Encoding qualitative variables...")
    df_encoded = encode_qualitative_variables(df_clean)
    
    # Step 4: Normalize data
    print("\n4. Normalizing data (z-score)...")
    numeric_cols = [
        'Urban_Pop_2024', 'Agri_Land_2000', 'Poverty_365_Year',
        'Gini_Index_Year', 'Historical_Temp_Trend',
        'Projected_Temp_Change_2050', 'Sea_Level_Rise_2050'
    ]
    
    df_normalized, scaler = normalize_data(df_encoded, numeric_cols)
    
    # Step 5: Perform clustering
    print("\n5. Performing clustering analysis...")
    df_clustered, kmeans, dbscan = perform_clustering(df_normalized)
    
    # Map cluster labels
    cluster_names = {
        0: 'High Poverty & Compound Hazards',
        1: 'High-Exposure Coastal',
        2: 'Middle-Income & Diversified Risk'
    }
    
    df_clustered['Cluster_Label'] = df_clustered['KMeans_Cluster'].map(cluster_names)
    
    # Step 6: Calculate CVS
    print("\n6. Calculating Compound Vulnerability Score...")
    df_cvs = calculate_cvs(df_clustered)
    
    # Step 7: Train XGBoost model
    print("\n7. Training XGBoost prediction model...")
    xgb_model, feature_importance = train_xgboost_model(df_cvs)
    
    # Step 8: Predict future scenario
    print("\n8. Predicting future vulnerability (High Stress 2050)...")
    df_future = predict_future_scenario(df_cvs, xgb_model, poverty_increase=0.15)
    
    # Step 9: Save all results
    print("\n9. Saving results...")
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter('climate_vulnerability_analysis_results.xlsx') as writer:
        df_raw.to_excel(writer, sheet_name='1_Raw_Data', index=False)
        df_clean.to_excel(writer, sheet_name='2_Cleaned_Data', index=False)
        df_encoded.to_excel(writer, sheet_name='3_Encoded_Data', index=False)
        df_normalized.to_excel(writer, sheet_name='4_Normalized_Data', index=False)
        df_clustered[['Country', 'KMeans_Cluster', 'Cluster_Label', 'DBSCAN_Cluster']].to_excel(
            writer, sheet_name='5_Clustering_Results', index=False)
        df_cvs[['Country', 'Exposure_Score', 'Sensitivity_Score', 'Trend_Score', 
                'CVS_Raw', 'CVS_Normalized']].to_excel(
            writer, sheet_name='6_CVS_Results', index=False)
        feature_importance.to_excel(writer, sheet_name='7_Feature_Importance', index=False)
        df_future[['Country', 'CVS_Normalized', 'CVS_Future_Predicted', 'CVS_Change']].to_excel(
            writer, sheet_name='8_Future_Predictions', index=False)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Display key results
    print("\nKEY RESULTS:")
    print("-" * 40)
    
    # Clustering results
    print("\nCLUSTER ASSIGNMENTS (K-Means):")
    cluster_summary = df_clustered[['Country', 'Cluster_Label', 'DBSCAN_Cluster']].copy()
    cluster_summary['DBSCAN_Label'] = cluster_summary['DBSCAN_Cluster'].apply(
        lambda x: 'Outlier' if x == -1 else f'Cluster {x}')
    print(cluster_summary.to_string(index=False))
    
    # Vulnerability scores
    print("\nCOMPOUND VULNERABILITY SCORES (0-100):")
    vuln_summary = df_cvs[['Country', 'CVS_Normalized']].copy()
    vuln_summary['CVS_Normalized'] = vuln_summary['CVS_Normalized'].round(1)
    vuln_summary = vuln_summary.sort_values('CVS_Normalized', ascending=False)
    print(vuln_summary.to_string(index=False))
    
    # Future predictions
    print("\nFUTURE VULNERABILITY PREDICTIONS (2050 High Stress):")
    future_summary = df_future[['Country', 'CVS_Normalized', 'CVS_Future_Predicted', 'CVS_Change']].copy()
    future_summary = future_summary.round(1)
    future_summary = future_summary.sort_values('CVS_Change', ascending=False)
    print(future_summary.to_string(index=False))
    
    print(f"\nAll results saved to: climate_vulnerability_analysis_results.xlsx")
    
    return df_future

# ============================================
# EXECUTE ANALYSIS
# ============================================

if __name__ == "__main__":
    results_df = main()
