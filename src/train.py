"""
Model Training Pipeline for Churn Prediction
File: src/train.py

Purpose: Train and compare ML models for churn prediction
- Logistic Regression (Baseline, interpretable)
- XGBoost (Production model, high performance)
- Proper train/test split
- Handle class imbalance
- Hyperparameter tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    """
    Train and evaluate churn prediction models.
    
    Models:
    1. Logistic Regression - Baseline, interpretable
    2. XGBoost - Production model
    
    Evaluation Metrics:
    - ROC-AUC (overall discrimination)
    - Recall (catch churners)
    - Precision (avoid false alarms)
    - F1-Score (balance)
    """
    
    def __init__(self, features_path='data/processed/churn_features.csv'):
        """
        Initialize trainer.
        
        Args:
            features_path (str): Path to engineered features
        """
        self.features_path = Path(features_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load engineered features"""
        print("="*70)
        print("📂 LOADING FEATURE DATA")
        print("="*70)
        
        self.df = pd.read_csv(self.features_path)
        print(f"✅ Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        print(f"   Churn Rate: {self.df['Churn'].mean()*100:.2f}%")
        
        return self
    
    def prepare_features(self):
        """
        Prepare features for modeling.
        
        Steps:
        1. Select features (remove non-numeric, target)
        2. Handle categorical variables
        3. Create feature list
        """
        print("\n" + "="*70)
        print("🔧 PREPARING FEATURES FOR MODELING")
        print("="*70)
        
        # Remove target and non-predictive columns
        exclude_cols = ['Churn', 'TenureGroup', 'RiskCategory']
        
        # Get numeric features
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Handle categorical columns that need encoding
        categorical_cols = self.df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            print(f"\n   Encoding categorical columns: {categorical_cols}")
            for col in categorical_cols:
                # Get dummies and drop first to avoid multicollinearity
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                feature_cols.remove(col)
                feature_cols.extend(dummies.columns.tolist())
        
        # Final feature list
        self.feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        print(f"\n   ✓ Total features for modeling: {len(self.feature_cols)}")
        print(f"   ✓ Features: {', '.join(self.feature_cols[:10])}...")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            test_size (float): Proportion for test set
            random_state (int): Random seed
        """
        print("\n" + "="*70)
        print("📊 SPLITTING DATA")
        print("="*70)
        
        X = self.df[self.feature_cols]
        y = self.df['Churn']
        
        # Stratified split to maintain churn rate
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n   Train set: {len(self.X_train):,} samples")
        print(f"   Test set: {len(self.X_test):,} samples")
        print(f"   Train churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"   Test churn rate: {self.y_test.mean()*100:.2f}%")
        
        return self
    
    def scale_features(self):
        """
        Scale features using StandardScaler.
        Important for Logistic Regression, not for XGBoost.
        """
        print("\n   Scaling features...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   ✓ Features scaled (mean=0, std=1)")
        
        return self
    
    def handle_class_imbalance(self, method='smote'):
        """
        Handle class imbalance.
        
        Args:
            method (str): 'smote' or 'class_weights'
        """
        print("\n" + "="*70)
        print("⚖️ HANDLING CLASS IMBALANCE")
        print("="*70)
        
        print(f"\n   Method: {method.upper()}")
        
        if method == 'smote':
            print(f"   Before SMOTE: {len(self.X_train):,} samples")
            
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
            
            print(f"   After SMOTE: {len(self.X_train_balanced):,} samples")
            print(f"   New churn rate: {self.y_train_balanced.mean()*100:.2f}%")
            
            # For XGBoost, use original data
            self.X_train_xgb = self.X_train
            self.y_train_xgb = self.y_train
            
        else:  # class_weights
            print(f"   Using class weights in model")
            self.X_train_balanced = self.X_train_scaled
            self.y_train_balanced = self.y_train
            self.X_train_xgb = self.X_train
            self.y_train_xgb = self.y_train
        
        return self
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression baseline model.
        
        Why Logistic Regression:
        - Interpretable (feature coefficients)
        - Fast training
        - Good baseline for comparison
        - Works well with balanced data
        """
        print("\n" + "="*70)
        print("🎯 TRAINING LOGISTIC REGRESSION (BASELINE)")
        print("="*70)
        
        # Train with balanced data
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle imbalance
        )
        
        print("\n   Training...")
        lr_model.fit(self.X_train_balanced, self.y_train_balanced)
        print("   ✓ Training complete")
        
        # Predictions
        y_pred_train = lr_model.predict(self.X_train_balanced)
        y_pred_test = lr_model.predict(self.X_test_scaled)
        y_pred_proba_test = lr_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        train_score = roc_auc_score(self.y_train_balanced, 
                                     lr_model.predict_proba(self.X_train_balanced)[:, 1])
        test_score = roc_auc_score(self.y_test, y_pred_proba_test)
        
        print(f"\n   📊 Performance:")
        print(f"      Train ROC-AUC: {train_score:.4f}")
        print(f"      Test ROC-AUC: {test_score:.4f}")
        
        # Store model and results
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'model': lr_model,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba_test,
            'train_auc': train_score,
            'test_auc': test_score
        }
        
        return self
    
    def train_xgboost(self):
        """
        Train XGBoost production model.
        
        Why XGBoost:
        - Handles non-linear relationships
        - Feature importance
        - Strong performance on tabular data
        - Built-in regularization
        """
        print("\n" + "="*70)
        print("🚀 TRAINING XGBOOST (PRODUCTION MODEL)")
        print("="*70)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (self.y_train_xgb == 0).sum() / (self.y_train_xgb == 1).sum()
        
        # XGBoost parameters
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'early_stopping_rounds': 20
        }
        
        print(f"\n   Parameters:")
        print(f"      scale_pos_weight: {scale_pos_weight:.2f} (handles imbalance)")
        print(f"      max_depth: {xgb_params['max_depth']}")
        print(f"      learning_rate: {xgb_params['learning_rate']}")
        print(f"      n_estimators: {xgb_params['n_estimators']}")
        
        # Train with early stopping
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        print("\n   Training with early stopping...")
        xgb_model.fit(
            self.X_train_xgb, 
            self.y_train_xgb,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        print(f"   ✓ Training complete (best iteration: {xgb_model.best_iteration})")
        
        # Predictions
        y_pred_train = xgb_model.predict(self.X_train_xgb)
        y_pred_test = xgb_model.predict(self.X_test)
        y_pred_proba_train = xgb_model.predict_proba(self.X_train_xgb)[:, 1]
        y_pred_proba_test = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        train_score = roc_auc_score(self.y_train_xgb, y_pred_proba_train)
        test_score = roc_auc_score(self.y_test, y_pred_proba_test)
        
        print(f"\n   📊 Performance:")
        print(f"      Train ROC-AUC: {train_score:.4f}")
        print(f"      Test ROC-AUC: {test_score:.4f}")
        
        # Store model and results
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'model': xgb_model,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba_test,
            'train_auc': train_score,
            'test_auc': test_score
        }
        
        return self
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation.
        
        Metrics:
        - ROC-AUC
        - Precision, Recall, F1
        - Confusion Matrix
        """
        print("\n" + "="*70)
        print("📈 MODEL EVALUATION")
        print("="*70)
        
        for model_name, results in self.results.items():
            print(f"\n{'='*70}")
            print(f"{model_name.upper().replace('_', ' ')}")
            print(f"{'='*70}")
            
            y_pred = results['y_pred']
            y_pred_proba = results['y_pred_proba']
            
            # Metrics
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"\n📊 Metrics:")
            print(f"   ROC-AUC:   {results['test_auc']:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\n📋 Confusion Matrix:")
            print(f"                Predicted")
            print(f"                No    Yes")
            print(f"   Actual No  {cm[0,0]:5d} {cm[0,1]:5d}")
            print(f"   Actual Yes {cm[1,0]:5d} {cm[1,1]:5d}")
            
            # Store detailed results
            self.results[model_name].update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            })
        
        return self
    
    def compare_models(self):
        """Compare models side by side"""
        print("\n" + "="*70)
        print("🔄 MODEL COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<15} {'Logistic Reg':<15} {'XGBoost':<15} {'Winner':<15}")
        print("-"*70)
        
        lr = self.results['logistic_regression']
        xgb_res = self.results['xgboost']
        
        metrics = [
            ('ROC-AUC', 'test_auc'),
            ('Recall', 'recall'),
            ('Precision', 'precision'),
            ('F1-Score', 'f1')
        ]
        
        for metric_name, metric_key in metrics:
            lr_val = lr[metric_key]
            xgb_val = xgb_res[metric_key]
            winner = 'XGBoost' if xgb_val > lr_val else 'Logistic Reg'
            
            print(f"{metric_name:<15} {lr_val:<15.4f} {xgb_val:<15.4f} {winner:<15}")
        
        return self
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        print("\n" + "="*70)
        print("💾 SAVING MODELS")
        print("="*70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_dir / f'{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✓ Saved: {model_path}")
        
        # Save scaler
        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ Saved: {scaler_path}")
        
        # Save feature names
        feature_path = output_dir / 'feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_cols, f, indent=2)
        print(f"   ✓ Saved: {feature_path}")
        
        return self
    
    def train_pipeline(self):
        """Execute complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING MODEL TRAINING PIPELINE")
        print("="*70)
        
        self.load_data()
        self.prepare_features()
        self.split_data()
        self.scale_features()
        self.handle_class_imbalance(method='smote')
        self.train_logistic_regression()
        self.train_xgboost()
        self.evaluate_models()
        self.compare_models()
        self.save_models()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print("\n🎯 Next Steps:")
        print("   1. Review model performance above")
        print("   2. Run src/evaluate.py for detailed analysis")
        print("   3. Use models for predictions")
        print("="*70 + "\n")
        
        return self


def main():
    """Main execution"""
    trainer = ChurnModelTrainer()
    trainer.train_pipeline()


if __name__ == "__main__":
    main()