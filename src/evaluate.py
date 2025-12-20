"""
Model Evaluation and Business Analysis
File: src/evaluate.py

Purpose: Comprehensive model evaluation including:
- Performance metrics
- Business cost analysis
- Feature importance
- Prediction insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')


class ChurnModelEvaluator:
    """
    Comprehensive evaluation of churn prediction models.
    
    Features:
    - Performance metrics
    - Business cost analysis (critical for interviews!)
    - ROC and PR curves
    - Feature importance
    - Prediction examples
    """
    
    def __init__(self, models_dir='models', data_path='data/processed/churn_features.csv'):
        """
        Initialize evaluator.
        
        Args:
            models_dir (str): Directory containing saved models
            data_path (str): Path to feature data
        """
        self.models_dir = Path(models_dir)
        self.data_path = Path(data_path)
        self.models = {}
        self.results = {}
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        print("="*70)
        print("📂 LOADING MODELS AND DATA")
        print("="*70)
        
        # Load models
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"   ✓ Loaded: {model_name}")
        
        # Load scaler
        with open(self.models_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"   ✓ Loaded: scaler")
        
        # Load feature names
        with open(self.models_dir / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        print(f"   ✓ Loaded: feature names ({len(self.feature_names)} features)")
        
        # Load data (we'll use the full dataset and split again for consistency)
        self.df = pd.read_csv(self.data_path)
        print(f"   ✓ Loaded: data ({len(self.df):,} records)")
        
        return self
    
    def business_cost_analysis(self, fn_cost=5000, fp_cost=500):
        """
        Calculate business cost for different models.
        
        THIS IS GOLD FOR INTERVIEWS!
        
        Costs:
        - False Negative (missed churner): ₹5,000 (lost customer)
        - False Positive (unnecessary offer): ₹500 (retention offer cost)
        
        Args:
            fn_cost (int): Cost of false negative
            fp_cost (int): Cost of false positive
        """
        print("\n" + "="*70)
        print("💰 BUSINESS COST ANALYSIS")
        print("="*70)
        
        print(f"\n💵 Cost Assumptions:")
        print(f"   False Negative (missed churner): ₹{fn_cost:,}")
        print(f"   False Positive (wrong prediction): ₹{fp_cost:,}")
        print(f"   Rationale: Acquiring new customer costs 10x retention offer")
        
        # For each model, calculate costs
        from sklearn.model_selection import train_test_split
        
        X = self.df[self.feature_names]
        y = self.df['Churn']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\n📊 Cost Analysis per Model:\n")
        print(f"{'Model':<20} {'FN':<8} {'FP':<8} {'Total Cost':<15} {'Cost/Customer':<15}")
        print("-"*70)
        
        for model_name, model in self.models.items():
            # Make predictions
            if model_name == 'logistic_regression':
                X_test_proc = self.scaler.transform(X_test)
            else:
                X_test_proc = X_test
            
            y_pred = model.predict(X_test_proc)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate costs
            fn_total = fn * fn_cost
            fp_total = fp * fp_cost
            total_cost = fn_total + fp_total
            cost_per_customer = total_cost / len(y_test)
            
            print(f"{model_name:<20} {fn:<8} {fp:<8} ₹{total_cost:<14,} ₹{cost_per_customer:<14.2f}")
            
            # Store results
            self.results[model_name] = {
                'fn': fn,
                'fp': fp,
                'total_cost': total_cost,
                'cost_per_customer': cost_per_customer
            }
        
        # Calculate savings vs do-nothing approach
        total_churners = y_test.sum()
        do_nothing_cost = total_churners * fn_cost
        
        print("\n" + "-"*70)
        print(f"{'Do Nothing (baseline)':<20} {total_churners:<8} {0:<8} ₹{do_nothing_cost:<14,} ₹{do_nothing_cost/len(y_test):<14.2f}")
        
        # Show savings
        print(f"\n💡 Cost Savings vs Do-Nothing:")
        for model_name, results in self.results.items():
            savings = do_nothing_cost - results['total_cost']
            savings_pct = (savings / do_nothing_cost) * 100
            print(f"   {model_name}: ₹{savings:,} ({savings_pct:.1f}% reduction)")
        
        return self
    
    def generate_performance_report(self):
        """Generate detailed performance report"""
        print("\n" + "="*70)
        print("📈 DETAILED PERFORMANCE REPORT")
        print("="*70)
        
        from sklearn.model_selection import train_test_split
        
        X = self.df[self.feature_names]
        y = self.df['Churn']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"{model_name.upper().replace('_', ' ')}")
            print(f"{'='*70}")
            
            # Prepare data
            if model_name == 'logistic_regression':
                X_test_proc = self.scaler.transform(X_test)
            else:
                X_test_proc = X_test
            
            y_pred = model.predict(X_test_proc)
            y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
            
            # Classification report
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
            
            # ROC-AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC Score: {auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n📋 Confusion Matrix:")
            print(f"                Predicted")
            print(f"                No Churn    Churn")
            print(f"Actual No Churn    {cm[0,0]:5d}    {cm[0,1]:5d}")
            print(f"Actual Churn       {cm[1,0]:5d}    {cm[1,1]:5d}")
            
            # Business metrics
            tn, fp, fn, tp = cm.ravel()
            print(f"\n💼 Business Metrics:")
            print(f"   True Positives (caught churners): {tp}")
            print(f"   False Negatives (missed churners): {fn}")
            print(f"   False Positives (unnecessary offers): {fp}")
            print(f"   Churn Capture Rate: {tp/(tp+fn)*100:.1f}%")
        
        return self
    
    def feature_importance_analysis(self):
        """Analyze and display feature importance"""
        print("\n" + "="*70)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            model = self.models['xgboost']
            importance = model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\n📊 Top 10 Most Important Features (XGBoost):\n")
            print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
            print("-"*50)
            
            for i, row in importance_df.head(10).iterrows():
                print(f"{i+1:<6} {row['Feature']:<30} {row['Importance']:<12.4f}")
            
            # Save to file
            importance_df.to_csv('reports/feature_importance.csv', index=False)
            print(f"\n   ✓ Full feature importance saved to: reports/feature_importance.csv")
        
        return self
    
    def prediction_examples(self, n_examples=5):
        """Show prediction examples for different risk levels"""
        print("\n" + "="*70)
        print("🎯 PREDICTION EXAMPLES")
        print("="*70)
        
        from sklearn.model_selection import train_test_split
        
        X = self.df[self.feature_names]
        y = self.df['Churn']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Use XGBoost for predictions
        model = self.models['xgboost']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get high risk, medium risk, low risk examples
        test_df = X_test.copy()
        test_df['Actual_Churn'] = y_test.values
        test_df['Churn_Probability'] = y_pred_proba
        test_df['Risk_Level'] = pd.cut(y_pred_proba, bins=[0, 0.3, 0.7, 1.0], 
                                        labels=['Low', 'Medium', 'High'])
        
        print(f"\n📋 Sample Predictions by Risk Level:\n")
        
        for risk_level in ['High', 'Medium', 'Low']:
            risk_examples = test_df[test_df['Risk_Level'] == risk_level].head(2)
            
            print(f"\n{risk_level} Risk Customers:")
            print("-"*70)
            
            for idx, row in risk_examples.iterrows():
                actual = "CHURNED" if row['Actual_Churn'] == 1 else "RETAINED"
                print(f"   Customer: Probability={row['Churn_Probability']:.2%}, Actual={actual}")
                
                # Show key features
                if 'tenure' in test_df.columns:
                    print(f"      Tenure: {row.get('tenure', 'N/A')} months")
                if 'ServiceCount' in test_df.columns:
                    print(f"      Services: {row.get('ServiceCount', 'N/A')}")
                if 'IsMonthToMonth' in test_df.columns:
                    print(f"      Month-to-Month: {'Yes' if row.get('IsMonthToMonth', 0) == 1 else 'No'}")
        
        return self
    
    def threshold_optimization(self):
        """
        Optimize prediction threshold based on business cost.
        
        Default threshold is 0.5, but we can optimize based on cost function.
        """
        print("\n" + "="*70)
        print("⚙️ THRESHOLD OPTIMIZATION")
        print("="*70)
        
        from sklearn.model_selection import train_test_split
        
        X = self.df[self.feature_names]
        y = self.df['Churn']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Use XGBoost
        model = self.models['xgboost']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        fn_cost, fp_cost = 5000, 500
        
        best_threshold = 0.5
        best_cost = float('inf')
        
        print(f"\n📊 Testing thresholds from 0.1 to 0.9:\n")
        print(f"{'Threshold':<12} {'Total Cost':<15} {'FN':<8} {'FP':<8}")
        print("-"*50)
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            total_cost = (fn * fn_cost) + (fp * fp_cost)
            
            if threshold in [0.3, 0.5, 0.7]:  # Show key thresholds
                print(f"{threshold:<12.2f} ₹{total_cost:<14,} {fn:<8} {fp:<8}")
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        print(f"\n💡 Optimal Threshold: {best_threshold:.2f}")
        print(f"   Minimizes total cost at: ₹{best_cost:,}")
        print(f"   (vs default 0.5 threshold)")
        
        return self
    
    def save_evaluation_report(self):
        """Save comprehensive evaluation report"""
        print("\n" + "="*70)
        print("💾 SAVING EVALUATION REPORT")
        print("="*70)
        
        report_path = Path('reports/model_performance.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHURN PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("📊 BUSINESS COST ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(f"Cost Assumptions:\n")
            f.write(f"  - False Negative: ₹5,000 (lost customer)\n")
            f.write(f"  - False Positive: ₹500 (retention offer)\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Total Cost: ₹{results['total_cost']:,}\n")
                f.write(f"  Cost per Customer: ₹{results['cost_per_customer']:.2f}\n")
                f.write(f"  False Negatives: {results['fn']}\n")
                f.write(f"  False Positives: {results['fp']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("-"*70 + "\n")
            
            # Determine best model
            best_model = min(self.results.items(), key=lambda x: x[1]['total_cost'])
            f.write(f"Best Model: {best_model[0].upper()}\n")
            f.write(f"Reason: Lowest total business cost\n")
            f.write(f"\nNext Steps:\n")
            f.write(f"1. Deploy {best_model[0]} to production\n")
            f.write(f"2. Set up monitoring for model drift\n")
            f.write(f"3. Implement retention campaigns for high-risk customers\n")
        
        print(f"   ✓ Report saved to: {report_path}")
        
        return self
    
    def evaluate_pipeline(self):
        """Execute complete evaluation pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING EVALUATION PIPELINE")
        print("="*70)
        
        self.load_models_and_data()
        self.business_cost_analysis()
        self.generate_performance_report()
        self.feature_importance_analysis()
        self.prediction_examples()
        self.threshold_optimization()
        self.save_evaluation_report()
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        print("\n🎯 Key Takeaways:")
        print("   1. XGBoost typically performs best on business metrics")
        print("   2. Check reports/model_performance.txt for full analysis")
        print("   3. Use threshold optimization for production deployment")
        print("="*70 + "\n")


def main():
    """Main execution"""
    evaluator = ChurnModelEvaluator()
    evaluator.evaluate_pipeline()


if __name__ == "__main__":
    main()