#!/usr/bin/env python3
"""
AI Data Analysis Agent - ULTIMATE+++ (3D + Interactive + Multi-Dimensional + Chat)
Adds advanced interactive & 3D visualization, correlation explorer, multi-dimensional analytics,
advanced statistical plots, and a built‑in chat Q&A interface that can also run quick predictions.

Usage examples:
  python run_ai_analyst_ultimate.py --file sales_data.csv --target revenue
  python run_ai_analyst_ultimate.py --file my.csv --chat
  python run_ai_analyst_ultimate.py --file my.csv --target churn_risk --chat

Optional Streamlit chat UI:
  streamlit run run_ai_analyst_ultimate.py -- --file my.csv --target churn_risk --ui

Outputs include:
• 3D Visualizations
• Interactive Correlation Explorer
• Multi-dimensional Analysis
• Advanced Statistical Plots
• Interactive Chat Interface
"""
from __future__ import annotations

import os
import re
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Matplotlib for static PNGs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional aesthetics (not required for core features)
import seaborn as sns  # noqa: F401

from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Plotly for interactive HTMLs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")
pio.renderers.default = "plotly_mimetype"  # we'll save to HTML files

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Core Agent
# ──────────────────────────────────────────────────────────────────────────────
class UltimateDataAnalysisAgent:
    """Ultimate AI Data Analysis Agent with Advanced Features"""

    def __init__(self, name="Ultimate AI Analyst"):
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        self.plot_count = 0
        self.output_dir: Optional[str] = None
        self.dataset_name = "unknown_dataset"
        self.using_sample_data = False

        # Advanced analysis components
        self.time_series_columns: List[str] = []
        self.ml_models: Dict[str, Any] = {}
        self.insights: List[str] = []
        self.anomalies: List[Any] = []

    # ── Setup & data ────────────────────────────────────────────────────────
    def _create_output_directory(self, dataset_name):
        clean_name = "".join(c for c in dataset_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        clean_name = clean_name.replace(" ", "_")[:50]
        self.dataset_name = clean_name if clean_name else "unknown_dataset"
        self.output_dir = f"ultimate_analysis_{self.dataset_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}/")
        return self.output_dir

    def load_data(self, data_source):
        dataset_name = "sample_data"
        self.using_sample_data = False

        if data_source is None:
            print("❌ No data source provided. Using sample data for demonstration.")
            self.data = self._create_enhanced_sample_data()
            dataset_name = "sample_data"
            self.using_sample_data = True
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source
            dataset_name = "dataframe_input"
            print("✅ Loaded data from DataFrame")
        elif isinstance(data_source, str) and os.path.exists(data_source):
            try:
                print(f"📂 Loading: {data_source}")
                if data_source.endswith(".csv"):
                    self.data = pd.read_csv(data_source)
                elif data_source.endswith(".xlsx"):
                    self.data = pd.read_excel(data_source)
                elif data_source.endswith(".json"):
                    self.data = pd.read_json(data_source)
                else:
                    raise ValueError("Unsupported file format")
                dataset_name = os.path.splitext(os.path.basename(data_source))[0]
                print(f"✅ Successfully loaded: {data_source}")
            except Exception as e:
                print(f"❌ Error loading file: {e}\n📊 Using sample data instead")
                self.data = self._create_enhanced_sample_data()
                dataset_name = "sample_data"
                self.using_sample_data = True
        else:
            print("❌ Invalid data source -> using sample data")
            self.data = self._create_enhanced_sample_data()
            dataset_name = "sample_data"
            self.using_sample_data = True

        self._create_output_directory(dataset_name)
        self._auto_detect_time_series()
        print(f"✅ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"📊 Columns: {list(self.data.columns)}")
        print("🔍 USING SAMPLE DATA FOR DEMONSTRATION" if self.using_sample_data else "🔍 USING YOUR CUSTOM DATASET")
        return self.data

    def _create_enhanced_sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
        
        # Create sample data that matches Streamlit app expectations
        sample_data = pd.DataFrame({
            'date': dates,
            'sales': np.random.gamma(2, 10000, n_samples) + 5000,
            'marketing_spend': np.random.normal(5000, 2000, n_samples),
            'website_visits': np.random.poisson(2000, n_samples),
            'customer_rating': np.random.uniform(3, 5, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples),
            'promotion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        })
        
        # Add realistic patterns to match Streamlit app
        sample_data['sales'] = sample_data['sales'] * (
            1 + 0.001 * np.arange(n_samples) + 
            0.2 * sample_data['promotion'] + 
            0.0001 * sample_data['marketing_spend'] +
            0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        )
        
        # Add additional columns for comprehensive analysis
        sample_data['customer_id'] = range(1, n_samples + 1)
        sample_data['age'] = np.random.normal(35, 10, n_samples)
        sample_data['income'] = np.random.normal(50000, 15000, n_samples)
        sample_data['purchase_amount'] = np.random.gamma(2, 50, n_samples)
        sample_data['purchase_frequency'] = np.random.poisson(3, n_samples)
        sample_data['loyalty_months'] = np.random.exponential(12, n_samples)
        sample_data['email_optin'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        sample_data['churn_risk'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        sample_data['customer_satisfaction'] = np.random.randint(1, 11, n_samples)
        
        print("📊 ENHANCED SAMPLE DATA CREATED (Streamlit Compatible)")
        return sample_data

    def _auto_detect_time_series(self):
        self.time_series_columns = []
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                try:
                    pd.to_datetime(self.data[col])
                    self.time_series_columns.append(col)
                except Exception:
                    pass
            elif "date" in col.lower() or "time" in col.lower():
                self.time_series_columns.append(col)
        print(f"⏰ Auto-detected time series columns: {self.time_series_columns}")

    # ── Time series ──────────────────────────────────────────────────────────
    def run_time_series_analysis(self):
        if not self.time_series_columns:
            print("⏰ No time series columns detected")
            return {}
        results = {}
        for ts_col in self.time_series_columns:
            try:
                self.data[ts_col] = pd.to_datetime(self.data[ts_col])
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                for num_col in numeric_cols[:3]:
                    ts_results = self._analyze_single_time_series(ts_col, num_col)
                    results[f"{ts_col}_{num_col}"] = ts_results
            except Exception as e:
                print(f"❌ Error in time series analysis for {ts_col}: {e}")
        return results

    def _analyze_single_time_series(self, date_col, value_col):
        ts_data = self.data[[date_col, value_col]].sort_values(date_col).dropna()
        if len(ts_data) < 2:
            return {}
        results = {
            "trend": self._calculate_trend(ts_data, date_col, value_col),
            "seasonality": self._detect_seasonality(ts_data, value_col),
            "stationarity": self._check_stationarity(ts_data[value_col]),
            "forecast": self._simple_forecast(ts_data, date_col, value_col),
            "anomalies": self._detect_anomalies(ts_data[value_col]),
        }
        self._create_time_series_plots(ts_data, date_col, value_col, results)
        return results

    def _calculate_trend(self, data, date_col, value_col):
        x = np.arange(len(data))
        y = data[value_col].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.5 else "weak",
        }

    def _detect_seasonality(self, data, value_col):
        from scipy import fftpack

        values = data[value_col].values
        n = len(values)
        if n < 10:
            return {"has_seasonality": False, "period": None}
        detrended = values - np.polyval(np.polyfit(np.arange(n), values, 1), np.arange(n))
        fft = fftpack.fft(detrended)
        frequencies = fftpack.fftfreq(n)
        power = np.abs(fft)
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = frequencies[dominant_idx]
        if abs(dominant_freq) > 0:
            period = int(1 / abs(dominant_freq))
            has_seasonality = power[dominant_idx] > np.mean(power) * 2
        else:
            period = None
            has_seasonality = False
        return {"has_seasonality": bool(has_seasonality), "period": period, "dominant_frequency": float(dominant_freq)}

    def _check_stationarity(self, series):
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore
            result = adfuller(series.dropna())
            return {"is_stationary": result[1] <= 0.05, "p_value": float(result[1]), "test_statistic": float(result[0])}
        except Exception:
            return {"is_stationary": False, "p_value": 1.0}

    def _simple_forecast(self, data, date_col, value_col, periods=30):
        x = np.arange(len(data))
        y = data[value_col].values
        slope, intercept, *_ = stats.linregress(x, y)
        future_x = np.arange(len(data), len(data) + periods)
        future_dates = pd.date_range(start=data[date_col].iloc[-1], periods=periods + 1, freq="D")[1:]
        future_values = intercept + slope * future_x
        return {
            "future_dates": future_dates.astype(str).tolist(),
            "forecasted_values": [float(v) for v in future_values],
            "confidence_interval": float(np.std(y) * 1.96),
        }

    def _detect_anomalies(self, series, method="iqr"):
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = series[(series < lower_bound) | (series > upper_bound)]
        return {
            "anomaly_count": int(len(anomalies)),
            "anomaly_indices": list(map(int, anomalies.index.tolist())),
            "anomaly_values": [float(v) for v in anomalies.values.tolist()],
        }

    def _create_time_series_plots(self, data, date_col, value_col, analysis_results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(data[date_col], data[value_col], label="Actual")
        axes[0, 0].set_title(f"Time Series: {value_col}")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel(value_col)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        x = np.arange(len(data))
        trend = analysis_results["trend"]["intercept"] + analysis_results["trend"]["slope"] * x
        axes[0, 1].plot(data[date_col], data[value_col], alpha=0.7, label="Actual")
        axes[0, 1].plot(data[date_col], trend, "r-", label="Trend")
        axes[0, 1].set_title(f"Trend Analysis: {value_col}")
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis="x", rotation=45)

        if len(data) > 50:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore

                decomposition = seasonal_decompose(data[value_col], period=min(30, len(data) // 2), model="additive")
                axes[1, 0].plot(decomposition.trend)
                axes[1, 0].set_title("Trend Component")
                axes[1, 1].plot(decomposition.seasonal)
                axes[1, 1].set_title("Seasonal Component")
            except Exception:
                axes[1, 0].hist(data[value_col], bins=20)
                axes[1, 0].set_title("Distribution")
                rolling_mean = data[value_col].rolling(window=7).mean()
                axes[1, 1].plot(data[date_col], rolling_mean)
                axes[1, 1].set_title("7-Day Rolling Mean")
                axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_plot(plt, f"timeseries_{value_col}")

    # ── AutoML ───────────────────────────────────────────────────────────────
    def run_automated_ml(self, target_column=None):
        if target_column is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[0]
                print(f"🤖 Auto-selected target column: {target_column}")
            else:
                print("❌ No suitable target column found for ML")
                return {}
        if target_column not in self.data.columns:
            print(f"❌ Target column '{target_column}' not found")
            return {}
        print(f"🤖 Running Automated ML for target: {target_column}")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        X_encoded = self._encode_categorical_features(X)
        X_encoded = X_encoded.dropna(axis=1, thresh=len(X_encoded) * 0.5)
        if X_encoded.shape[1] == 0:
            print("❌ Not enough features for ML after preprocessing")
            return {}

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_encoded)

        problem_type = self._determine_problem_type(y)
        print(f"🔍 Problem type: {problem_type}")
        ml_results = self._run_ml_pipeline(X_imputed, y, problem_type, target_column)
        if "feature_importance" in ml_results and ml_results["feature_importance"] is not None:
            self._plot_feature_importance(ml_results["feature_importance"], X_encoded.columns)
        self.ml_models[target_column] = ml_results
        self.ml_models[target_column]["feature_names"] = list(X_encoded.columns)
        self.ml_models[target_column]["imputer_strategy"] = "mean"
        return ml_results

    def _determine_problem_type(self, y):
        if y.dtype == "object" or len(y.unique()) < 10:
            return "classification"
        else:
            return "regression"

    def _encode_categorical_features(self, X):
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=["object"]).columns:
            if len(X_encoded[col].unique()) <= 10:
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
            else:
                X_encoded = X_encoded.drop(columns=[col])
        # convert datetimes to ordinal
        for col in X_encoded.columns:
            if np.issubdtype(X_encoded[col].dtype, np.datetime64):
                X_encoded[col] = X_encoded[col].map(lambda d: d.toordinal())
        return X_encoded

    def _run_ml_pipeline(self, X, y, problem_type, target_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {}
        performances = {}
        if problem_type == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
                "SVR": SVR(),
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
                "SVC": SVC(probability=True, random_state=42),
            }

        best_score = -np.inf
        best_model = None
        best_model_name = None

        from sklearn.metrics import r2_score, mean_absolute_error

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if problem_type == "regression":
                    score = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    performances[name] = {
                        "r2_score": float(score),
                        "mae": float(mae),
                        "rmse": rmse,
                        "cv_score": float(np.mean(cross_val_score(model, X, y, cv=5))),
                    }
                else:
                    score = accuracy_score(y_test, y_pred)
                    performances[name] = {
                        "accuracy": float(score),
                        "cv_score": float(np.mean(cross_val_score(model, X, y, cv=5))),
                        "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    }
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

        feature_importance = None
        if hasattr(best_model, "feature_importances_"):
            feature_importance = {
                "feature": [f"Feature_{i}" for i in range(X.shape[1])],
                "importance": best_model.feature_importances_,
            }
        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_score": float(best_score),
            "model_performance": performances,
            "problem_type": problem_type,
            "feature_importance": feature_importance,
        }

    def _plot_feature_importance(self, feature_importance, feature_names):
        if feature_importance is None:
            return
        features = feature_names if len(feature_importance["feature"]) == len(feature_names) else feature_importance["feature"]
        importance_df = (
            pd.DataFrame({"feature": features, "importance": feature_importance["importance"]})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        self._save_plot(plt, "feature_importance")

    # ── Customer analytics ───────────────────────────────────────────────────
    def run_customer_analytics(self):
        print("👥 Running Customer Analytics...")
        rfm_columns = {}
        for col in self.data.columns:
            col_lower = col.lower()
            if "amount" in col_lower or "price" in col_lower or "revenue" in col_lower:
                rfm_columns["monetary"] = col
            elif "date" in col_lower or "time" in col_lower:
                rfm_columns["recency"] = col
            elif "frequency" in col_lower or "count" in col_lower:
                rfm_columns["frequency"] = col
        if len(rfm_columns) >= 2:
            return self._perform_rfm_analysis(rfm_columns)
        else:
            return self._perform_customer_segmentation()

    def _perform_rfm_analysis(self, rfm_columns):
        print("📊 Performing RFM Analysis...")
        rfm_data = self.data.copy()
        if "recency" in rfm_columns:
            rfm_data["recency_days"] = (pd.Timestamp.now() - pd.to_datetime(rfm_data[rfm_columns["recency"]])).dt.days
        rfm_results = {"customer_segments": {}, "segmentation_rules": "Basic RFM segmentation applied", "segment_counts": {}}
        if "monetary" in rfm_columns and "recency" in rfm_columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(rfm_data[rfm_columns["recency"]], rfm_data[rfm_columns["monetary"]], alpha=0.6)
            plt.xlabel("Recency")
            plt.ylabel("Monetary Value")
            plt.title("Customer Value vs Recency")
            self._save_plot(plt, "rfm_analysis")
        return rfm_results

    def _perform_customer_segmentation(self):
        print("🎯 Performing Customer Segmentation...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric columns for segmentation"}
        segment_data = self.data[numeric_cols[:4]].dropna()
        if len(segment_data) < 10:
            return {"error": "Not enough data for segmentation"}
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(segment_data)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        optimal_clusters = self._find_elbow(wcss)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        self.data["customer_segment"] = clusters
        if segment_data.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(segment_data.iloc[:, 0], segment_data.iloc[:, 1], c=clusters, cmap="viridis")
            plt.xlabel(segment_data.columns[0])
            plt.ylabel(segment_data.columns[1])
            plt.title(f"Customer Segmentation ({optimal_clusters} Clusters)")
            plt.colorbar(scatter)
            self._save_plot(plt, "customer_segmentation")
        return {
            "n_clusters": int(optimal_clusters),
            "cluster_sizes": pd.Series(clusters).value_counts().to_dict(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
        }

    def _find_elbow(self, wcss):
        deltas = np.diff(wcss)
        deltas2 = np.diff(deltas)
        elbow = np.argmax(deltas2) + 2
        return int(min(max(2, elbow), 5))

    # ── Insights & BI ────────────────────────────────────────────────────────
    def generate_natural_language_insights(self):
        print("💬 Generating Natural Language Insights...")
        insights = []
        insights.append(f"The dataset contains {self.data.shape[0]:,} records with {self.data.shape[1]} features.")
        missing_total = self.data.isnull().sum().sum()
        if missing_total > 0:
            insights.append(f"⚠️ Data quality note: There are {int(missing_total):,} missing values that may need attention.")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"📈 Found {len(numeric_cols)} numeric features for analysis.")
            for col in numeric_cols[:3]:
                data = self.data[col].dropna()
                if len(data) > 0:
                    insights.append(f"• {col}: ranges from {data.min():.2f} to {data.max():.2f} with average {data.mean():.2f}")
        if self.time_series_columns:
            insights.append(f"⏰ Time series analysis available for {len(self.time_series_columns)} date columns.")
        if self.ml_models:
            for target, results in self.ml_models.items():
                best_model = results["best_model_name"]
                best_score = results["best_score"]
                insights.append(f"🤖 For predicting {target}, {best_model} achieved score of {best_score:.3f}")
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr))
            if strong_corrs:
                insights.append("🔗 Strong correlations found:")
                for col1, col2, corr in strong_corrs[:3]:
                    insights.append(f"  • {col1} ↔ {col2}: {corr:.3f}")
        self.insights = insights
        return insights

    def create_bi_dashboard_data(self):
        print("📊 Preparing BI Dashboard Data...")
        bi_data = {
            "summary_metrics": self._calculate_summary_metrics(),
            "kpi_trends": self._calculate_kpi_trends(),
            "performance_benchmarks": self._calculate_benchmarks(),
            "anomalies": self.anomalies,
            "recommendations": self._generate_recommendations(),
        }
        with open(f"{self.output_dir}/bi_dashboard_data.json", "w") as f:
            json.dump(bi_data, f, indent=2)
        return bi_data

    def _calculate_summary_metrics(self):
        metrics = {}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            data = self.data[col].dropna()
            if len(data) == 0:
                continue
            metrics[col] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "growth_rate": self._calculate_growth_rate(data) if len(data) > 1 else 0,
            }
        return metrics

    def _calculate_growth_rate(self, series):
        if len(series) < 2:
            return 0
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)

    def _calculate_kpi_trends(self):
        trends = {}
        if self.time_series_columns and len(self.data) > 10:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:
                try:
                    x = np.arange(len(self.data))
                    y = self.data[col].values
                    slope, _, r_value, _, _ = stats.linregress(x, y)
                    trends[col] = {"trend": "increasing" if slope > 0 else "decreasing", "strength": float(abs(r_value)), "slope": float(slope)}
                except Exception:
                    continue
        return trends

    def _calculate_benchmarks(self):
        benchmarks = {}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            data = self.data[col].dropna()
            if len(data) == 0:
                continue
            q75, q25 = np.percentile(data, [75, 25])
            benchmarks[col] = {"excellent": float(q75), "good": float(np.median(data)), "needs_improvement": float(q25)}
        return benchmarks

    def _generate_recommendations(self):
        recommendations = []
        missing_cols = self.data.isnull().sum()
        high_missing = missing_cols[missing_cols > len(self.data) * 0.1]
        if len(high_missing) > 0:
            recommendations.append(f"Consider imputing or removing columns with high missing values: {', '.join(high_missing.index[:3])}")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        rec = f"High correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]} - consider feature selection"
                        if rec not in recommendations:
                            recommendations.append(rec)
        if self.time_series_columns:
            recommendations.append("Time series data detected - consider forecasting and seasonality analysis")
        if len(numeric_cols) >= 3:
            recommendations.append("Sufficient features available for machine learning modeling")
        return recommendations[:5]

    def _save_plot(self, plt_obj, description=""):
        self.plot_count += 1
        filename = f"{self.output_dir}/plot_{self.plot_count:02d}_{description.replace(' ', '_')}.png"
        plt_obj.savefig(filename, dpi=100, bbox_inches="tight")
        plt_obj.close()
        print(f"   📊 Saved: {filename}")
        return filename

    # ── NEW: Next‑level interactive & 3D visualizations (moved to suite) ─────
    # handled by AdvancedVisualizationSuite below

    # ── Orchestrator & reporting ────────────────────────────────────────────
    def run_comprehensive_analysis(self, target_column=None):
        print(f"🚀 {self.name} - ULTIMATE+++ COMPREHENSIVE ANALYSIS\n" + "=" * 70)
        if self.data is None:
            self.load_data(None)

        analysis_results = {
            "time_series": self.run_time_series_analysis(),
            "automated_ml": self.run_automated_ml(target_column),
            "customer_analytics": self.run_customer_analytics(),
            "natural_language_insights": self.generate_natural_language_insights(),
            "bi_dashboard": self.create_bi_dashboard_data(),
        }

        # NEW: interactive & 3D visualizations + explorers
        viz = AdvancedVisualizationSuite(self)
        viz.create_3d_visualizations(target_column)
        viz.create_interactive_correlation_explorer()
        viz.create_multi_dimensional_analysis()
        viz.create_advanced_statistical_plots()

        self.save_ultimate_report(analysis_results)
        print("\n" + "=" * 70)
        print("🎉 ULTIMATE+++ ANALYSIS COMPLETED!")
        print("=" * 70)
        print("\n🔍 KEY INSIGHTS:")
        for insight in self.insights[:15]:
            print(f"  {insight}")
        print(f"\n📁 All outputs saved in: {self.output_dir}/")
        print(f"📊 Total static visualizations (PNG): {self.plot_count}")
        print(f"🧭 Interactive HTML charts created in: {self.output_dir}/")
        print(f"🤖 ML Models trained: {len(self.ml_models)}")
        print(f"💬 Natural language insights: {len(self.insights)}")
        return analysis_results

    def save_ultimate_report(self, analysis_results):
        report_file = f"{self.output_dir}/ULTIMATE_ANALYSIS_REPORT.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# ULTIMATE DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"**Dataset:** {self.dataset_name}\n")
            f.write(f"**Records:** {self.data.shape[0]:,}\n")
            f.write(f"**Features:** {self.data.shape[1]}\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 📊 Executive Summary\n\n")
            for insight in self.insights[:10]:
                f.write(f"- {insight}\n")
            f.write("\n## 🤖 Machine Learning Results\n\n")
            for target, results in self.ml_models.items():
                f.write(f"### Target: {target}\n")
                f.write(f"- Best Model: {results['best_model_name']}\n")
                f.write(f"- Best Score: {results['best_score']:.3f}\n")
                f.write(f"- Problem Type: {results['problem_type']}\n\n")
            f.write("\n## ⏰ Time Series Analysis\n\n")
            for ts_key, ts_results in analysis_results["time_series"].items():
                f.write(f"### {ts_key}\n")
                if "trend" in ts_results:
                    f.write(f"- Trend: {ts_results['trend']['trend_direction']} ({ts_results['trend']['trend_strength']})\n")
                if "seasonality" in ts_results:
                    f.write(f"- Seasonality: {ts_results['seasonality']['has_seasonality']}\n")
                f.write("\n")
            f.write("\n## 💡 Recommendations\n\n")
            for rec in analysis_results["bi_dashboard"]["recommendations"]:
                f.write(f"- {rec}\n")
            f.write(f"\n## 📈 Generated static visualizations (PNG): {self.plot_count}\n\n")
            plot_files = [f for f in _safe_listdir(self.output_dir) if f.startswith("plot_")]
            for plot_file in plot_files:
                f.write(f"- {plot_file}\n")
            f.write("\n## 🧭 Interactive HTML visualizations\n\n")
            for html_file in sorted([f for f in _safe_listdir(self.output_dir) if f.endswith('.html')]):
                f.write(f"- {html_file}\n")
        print(f"📄 Ultimate report saved: {report_file}")


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Advanced Visualization Suite
# ──────────────────────────────────────────────────────────────────────────────
class AdvancedVisualizationSuite:
    def __init__(self, agent: UltimateDataAnalysisAgent):
        self.agent = agent
        self.data = agent.data
        self.output_dir = agent.output_dir

    def create_3d_visualizations(self, target_column: Optional[str] = None):
        html_files = []
        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        # 3D scatter
        if len(numeric_cols) >= 3:
            color = None
            if "customer_segment" in self.data.columns:
                color = "customer_segment"
            elif target_column and target_column in self.data.columns:
                color = target_column
            fig = px.scatter_3d(
                self.data,
                x=numeric_cols[0],
                y=numeric_cols[1],
                z=numeric_cols[2],
                color=color,
                opacity=0.7,
                title="3D Data Distribution",
            )
            path = f"{self.output_dir}/viz_3d_scatter.html"
            pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
            html_files.append(path)
        # 3D PCA
        if len(numeric_cols) >= 4:
            clean_df = self.data[numeric_cols].dropna()
            if len(clean_df) > 1:
                pca = PCA(n_components=3)
                X_scaled = StandardScaler().fit_transform(clean_df)
                comps = pca.fit_transform(X_scaled)
                df_pca = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"], index=clean_df.index)
                if target_column and target_column in self.data.columns:
                    df_pca[target_column] = self.data.loc[clean_df.index, target_column]
                fig = px.scatter_3d(
                    df_pca,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color=target_column if target_column in df_pca.columns else None,
                    title="3D PCA Components",
                )
                path = f"{self.output_dir}/viz_3d_pca.html"
                pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
                html_files.append(path)
        # 3D Correlation Surface
        if len(numeric_cols) >= 3:
            corr = self.data[numeric_cols].corr().values
            fig = go.Figure(data=[go.Surface(z=corr)])
            fig.update_layout(
                title="3D Correlation Surface",
                scene=dict(xaxis_title="Feature Index", yaxis_title="Feature Index", zaxis_title="Correlation"),
            )
            path = f"{self.output_dir}/viz_3d_correlation_surface.html"
            pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
            html_files.append(path)
        # 3D Time Surface (date x category -> value)
        ts_cols = getattr(self.agent, "time_series_columns", [])
        if ts_cols:
            date_col = ts_cols[0]
            cat_cols = [c for c in self.data.columns if self.data[c].dtype == "object" and c != date_col]
            num_cols = numeric_cols
            if cat_cols and num_cols:
                cat = cat_cols[0]
                val = num_cols[0]
                df = self.data[[date_col, cat, val]].dropna()
                if len(df) > 0:
                    df[date_col] = pd.to_datetime(df[date_col])
                    grid = df.pivot_table(index=date_col, columns=cat, values=val, aggfunc="mean").fillna(method="ffill").fillna(method="bfill")
                    if grid.shape[1] >= 2 and grid.shape[0] >= 10:
                        Z = grid.values
                        fig = go.Figure(data=[go.Surface(z=Z)])
                        fig.update_layout(
                            title=f"3D Time Surface: {val} by {cat} over {date_col}",
                            scene=dict(xaxis_title="Time", yaxis_title=f"{cat}", zaxis_title=f"{val}"),
                        )
                        path = f"{self.output_dir}/viz_3d_time_surface.html"
                        pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
                        html_files.append(path)
        print("✅ 3D Visualizations created.")
        return html_files

    def create_interactive_correlation_explorer(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        corr_matrix = self.data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", title="Interactive Correlation Explorer")
        path = f"{self.output_dir}/interactive_correlation_explorer.html"
        pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
        print("✅ Interactive correlation explorer saved.")
        return path

    def create_multi_dimensional_analysis(self):
        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) >= 4:
            dims = numeric_cols[: min(6, len(numeric_cols))]
            fig = px.scatter_matrix(self.data, dimensions=dims, title="Multi-Dimensional Analysis Explorer")
            path = f"{self.output_dir}/multi_dimensional_analysis.html"
            pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
            print("✅ Multi-dimensional analysis visualization saved.")
            return path
        return None

    def create_advanced_statistical_plots(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Distribution of {col}", f"QQ Plot: {col}"))
            fig.add_trace(go.Histogram(x=self.data[col], nbinsx=40, name="Histogram"), row=1, col=1)
            quantiles = np.linspace(0.01, 0.99, 100)
            theoretical = stats.norm.ppf(quantiles)
            actual = np.quantile(self.data[col].dropna(), quantiles)
            fig.add_trace(go.Scatter(x=theoretical, y=actual, mode="markers", name="QQ Points"), row=1, col=2)
            path = f"{self.output_dir}/advanced_stat_{col}.html"
            pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
        print("✅ Advanced statistical plots generated.")


# ──────────────────────────────────────────────────────────────────────────────
# NEW: HTML Report Generator for Streamlit Compatibility
# ──────────────────────────────────────────────────────────────────────────────
class HTMLReportGenerator:
    def __init__(self, agent: UltimateDataAnalysisAgent):
        self.agent = agent
        self.data = agent.data
        self.output_dir = agent.output_dir

    def create_reports(self):
        """Create HTML reports that can be displayed in Streamlit"""
        html_files = []
        
        # 1. Create Data Overview HTML
        html_files.append(self._create_data_overview_html())
        
        # 2. Create Correlation Matrix HTML
        html_files.append(self._create_correlation_html())
        
        # 3. Create Distribution Plots HTML
        html_files.extend(self._create_distribution_htmls())
        
        # 4. Create Time Series HTML (if available)
        if self.agent.time_series_columns:
            html_files.extend(self._create_timeseries_htmls())
        
        print(f"✅ Created {len(html_files)} HTML reports for Streamlit")
        return html_files

    def _create_data_overview_html(self):
        """Create comprehensive data overview HTML"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Overview - {self.agent.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>📊 Data Overview</h1>
            
            <div class="section">
                <h2>Dataset Metrics</h2>
                <div class="metric-card">
                    <h3>Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns</h3>
                    <p>Memory: {self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB</p>
                    <p>Missing Values: {self.data.isnull().sum().sum():,}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Column Information</h2>
                <p><strong>Numeric Columns ({len(numeric_cols)}):</strong> {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}</p>
                <p><strong>Categorical Columns ({len(categorical_cols)}):</strong> {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}</p>
            </div>
        </body>
        </html>
        """
        
        path = f"{self.output_dir}/data_overview.html"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path

    def _create_correlation_html(self):
        """Create correlation matrix HTML"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
            
        corr_matrix = self.data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix", color_continuous_scale="RdBu_r")
        path = f"{self.output_dir}/correlation_matrix.html"
        pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
        return path

    def _create_distribution_htmls(self):
        """Create distribution plots for numeric columns"""
        html_files = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            fig = px.histogram(self.data, x=col, title=f"Distribution of {col}", marginal="box")
            path = f"{self.output_dir}/distribution_{col}.html"
            pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
            html_files.append(path)
            
        return html_files

    def _create_timeseries_htmls(self):
        """Create time series plots"""
        html_files = []
        ts_cols = self.agent.time_series_columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for ts_col in ts_cols[:1]:  # Use first time series column
            for num_col in numeric_cols[:2]:  # Use first 2 numeric columns
                try:
                    ts_data = self.data[[ts_col, num_col]].dropna()
                    if len(ts_data) > 1:
                        ts_data[ts_col] = pd.to_datetime(ts_data[ts_col])
                        ts_data = ts_data.sort_values(ts_col)
                        fig = px.line(ts_data, x=ts_col, y=num_col, title=f"Time Series: {num_col}")
                        path = f"{self.output_dir}/timeseries_{num_col}.html"
                        pio.write_html(fig, path, auto_open=False, include_plotlyjs="cdn")
                        html_files.append(path)
                except Exception as e:
                    print(f"Error creating timeseries plot for {num_col}: {e}")
                    
        return html_files


# ──────────────────────────────────────────────────────────────────────────────
# Chat Q&A over the dataset & models
# ──────────────────────────────────────────────────────────────────────────────
class ChatAnalyst:
    """Rule‑based chat layer that can answer common questions & run predictions."""

    def __init__(self, agent: UltimateDataAnalysisAgent, target_column: Optional[str] = None):
        self.agent = agent
        self.df = agent.data
        self.target = target_column

    def answer(self, q: str) -> str:
        q_l = q.lower().strip()
        # Help & examples
        if q_l in {"help", "?", "examples", "example"}:
            return (
                "Examples:\n"
                "  • mean income\n"
                "  • total purchase_amount\n"
                "  • unique values of region\n"
                "  • correlation between income and purchase_amount\n"
                "  • head 5\n"
                "  • best model\n"
                "  • feature importance\n"
                "  • predict target=churn_risk with age=30, income=62000, purchase_frequency=2\n"
            )
        # Head
        m = re.match(r"head\s*(\d+)?", q_l)
        if m:
            n = int(m.group(1) or 5)
            return self.df.head(n).to_string(index=False)
        # Mean / avg of column
        m = re.match(r"(mean|avg|average)\s+([\w\s]+)", q_l)
        if m:
            col = self._closest_column(m.group(2))
            if col and pd.api.types.is_numeric_dtype(self.df[col]):
                return f"Mean of {col}: {self.df[col].mean():.4f}"
            return f"Column '{m.group(2)}' not found or not numeric."
        # Sum / total
        m = re.match(r"(sum|total)\s+([\w\s]+)", q_l)
        if m:
            col = self._closest_column(m.group(2))
            if col and pd.api.types.is_numeric_dtype(self.df[col]):
                return f"Total {col}: {self.df[col].sum():.4f}"
            return f"Column '{m.group(2)}' not found or not numeric."
        # Unique values
        m = re.match(r"(unique|distinct)\s+(values\s+of\s+)?([\w\s]+)", q_l)
        if m:
            col = self._closest_column(m.group(3))
            if col:
                vals = self.df[col].dropna().unique()
                preview = ", ".join(map(str, vals[:20]))
                more = " (…truncated)" if len(vals) > 20 else ""
                return f"Unique values of {col} ({len(vals)}): {preview}{more}"
            return f"Column '{m.group(3)}' not found."
        # Correlation
        m = re.match(r"(corr|correlation)\s+(between\s+)?([\w\s]+)\s+(and|&)\s+([\w\s]+)", q_l)
        if m:
            c1 = self._closest_column(m.group(3))
            c2 = self._closest_column(m.group(5))
            if c1 and c2 and all(pd.api.types.is_numeric_dtype(self.df[c]) for c in [c1, c2]):
                corr = self.df[[c1, c2]].corr().iloc[0, 1]
                return f"Correlation between {c1} and {c2}: {corr:.4f}"
            return "Columns not found or not numeric."
        # Best model
        if "best model" in q_l or "model score" in q_l:
            if not self.agent.ml_models:
                return "No trained models yet. Run analysis first."
            summaries = []
            for target, res in self.agent.ml_models.items():
                summaries.append(f"Target {target}: {res['best_model_name']} (score={res['best_score']:.3f}, type={res['problem_type']})")
            return "\n".join(summaries)
        # Feature importance
        if "feature importance" in q_l:
            if not self.agent.ml_models:
                return "No trained models yet."
            # pick first model with importances
            for target, res in self.agent.ml_models.items():
                imp = res.get("feature_importance")
                names = res.get("feature_names")
                if imp is not None and names is not None:
                    order = np.argsort(imp["importance"])[::-1][:10]
                    items = [f"{names[i]}: {imp['importance'][i]:.4f}" for i in order]
                    return "Top features (approx):\n" + "\n".join(items)
            return "Active best model does not expose importances."
        # Predict pattern
        m = re.match(r"predict\s+(target=)?([\w\-]+)\s+with\s+(.+)$", q_l)
        if m:
            target = m.group(2)
            feats = self._parse_kv_pairs(m.group(3))
            return self._predict_from_text(target, feats)
        # Fallback: simple stats
        if q_l.startswith("describe") or q_l == "summary":
            return str(self.df.describe(include="all").transpose().head(20))
        return "I didn't catch that. Type 'help' for examples, or ask for mean/sum/correlation, 'feature importance', 'best model', or a 'predict … with …' query."

    def _closest_column(self, name: str) -> Optional[str]:
        name = name.strip()
        cols = list(self.df.columns)
        if name in cols:
            return name
        # fuzzy match by simple ratio
        scores = [(c, self._simple_ratio(name.lower(), c.lower())) for c in cols]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores and scores[0][1] >= 0.6 else None

    @staticmethod
    def _simple_ratio(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        inter = len(sa & sb)
        uni = len(sa | sb) or 1
        return inter / uni

    @staticmethod
    def _parse_kv_pairs(s: str) -> Dict[str, Any]:
        pairs = re.findall(r"([\w\-]+)\s*=\s*([^,]+)", s)
        out = {}
        for k, v in pairs:
            v = v.strip()
            try:
                if v.lower() in {"true", "false"}:
                    out[k] = v.lower() == "true"
                elif re.match(r"^-?\d+\.?\d*$", v):
                    out[k] = float(v)
                else:
                    out[k] = v
            except Exception:
                out[k] = v
        return out

    def _predict_from_text(self, target: str, features: Dict[str, Any]) -> str:
        if target not in self.agent.ml_models:
            return f"Model for target '{target}' not found. Available: {', '.join(self.agent.ml_models.keys()) or 'none'}"
        res = self.agent.ml_models[target]
        model = res["best_model"]
        feat_names = res.get("feature_names", [])
        # Build single-row input aligned to training features
        row = {f: np.nan for f in feat_names}
        for k, v in features.items():
            # try to map by closest column name
            ck = self._closest_column(k)
            if ck in feat_names:
                row[ck] = v
            elif k in feat_names:
                row[k] = v
        X = pd.DataFrame([row])[feat_names]
        # Impute missing with mean (as used in training)
        X = X.fillna(X.mean(numeric_only=True))
        try:
            yhat = model.predict(X)[0]
            if res["problem_type"] == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                probs = ", ".join(f"class_{i}:{p:.3f}" for i, p in enumerate(proba))
                return f"Prediction for {target}: {yhat} (probs: {probs})"
            return f"Prediction for {target}: {yhat}"
        except Exception as e:
            return f"Could not run prediction: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# CLI / Streamlit runner
# ──────────────────────────────────────────────────────────────────────────────
def run_cli(agent: UltimateDataAnalysisAgent, target: Optional[str]):
    print("\n💬 Chat mode: type questions (e.g., 'help'). Type 'exit' to quit.\n")
    chat = ChatAnalyst(agent, target)
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in {"exit", "quit", "q"}:
            print("👋 Bye!")
            break
        print("Agent:", chat.answer(q))


def run_streamlit_ui(agent: UltimateDataAnalysisAgent, target: Optional[str]):
    try:
        import streamlit as st
    except Exception:
        print("Streamlit not installed. Run 'pip install streamlit' or use --chat for CLI.")
        return

    st.set_page_config(page_title="AI Analyst ULTIMATE+++", layout="wide")
    st.title("🤖 AI Data Analysis Agent — ULTIMATE+++")
    st.caption("Interactive 3D viz, correlation explorer, multi‑dimensional analysis + Chat")

    # Left: list generated HTMLs
    with st.sidebar:
        st.markdown("### Outputs")
        if agent.output_dir:
            for f in sorted(_safe_listdir(agent.output_dir)):
                if f.endswith(".html"):
                    st.write(f"📄 {f}")
        st.markdown("---")
        st.markdown("**Tips**: ask 'mean income', 'correlation between age and income', or 'predict target=churn_risk with age=30, income=50000'.")

    chat = ChatAnalyst(agent, target)
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    for role, text in st.session_state.msgs:
        with st.chat_message(role):
            st.write(text)
    prompt = st.chat_input("Ask about the dataset or make a prediction…")
    if prompt:
        st.session_state.msgs.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)
        answer = chat.answer(prompt)
        st.session_state.msgs.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Data Analysis Agent — ULTIMATE+++")
    parser.add_argument("--file", default=None, help="Path to CSV/XLSX/JSON data file")
    parser.add_argument("--target", default=None, help="Target column for ML (optional)")
    parser.add_argument("--chat", action="store_true", help="Start CLI chat after analysis")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit chat UI (requires streamlit)")
    args = parser.parse_args()

    # If no file given, list all CSVs and ask user to choose one interactively
    if not args.file:
        import os
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if not files:
            print("❌ No CSV files found in the current directory!")
            return
        print("\n📂 Available CSV files:\n")
        for i, f in enumerate(files, start=1):
            print(f" {i}. {f}")
        choice = input("\n👉 Enter the number of the dataset you want to analyze: ")
        try:
            args.file = files[int(choice) - 1]
        except (IndexError, ValueError):
            print("❌ Invalid choice. Exiting.")
            return

    # Run the analysis
    agent = UltimateDataAnalysisAgent("ULTIMATE+++ AI Data Analysis Agent")
    agent.load_data(args.file)
    agent.run_comprehensive_analysis(args.target)

    # Automatically create and open the HTML dashboard
    HTMLReportGenerator(agent).create_reports()

    # Optional interactive or Streamlit modes
    if args.chat:
        run_cli(agent, args.target)
    if args.ui:
        run_streamlit_ui(agent, args.target)


if __name__ == "__main__":
    main()

import streamlit as st
from datetime import datetime

if 'chathistory' not in st.session_state:
    st.session_state['chathistory'] = []

userinput = st.text_input("Ask a question:")
if st.button("Send"):
    st.session_state['chathistory'].append({
        "role": "user",
        "content": userinput,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    # Imagine 'get_response' is your chat model:
    response = get_response(userinput)
    st.session_state['chathistory'].append({
        "role": "assistant",
        "content": response,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

for msg in st.session_state['chathistory']:
    st.markdown(f"**{msg['role']}**: {msg['content']} ({msg['time']})")
