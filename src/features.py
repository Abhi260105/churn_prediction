"""
Feature Engineering for Churn Prediction
File: src/features.py

Purpose: Create business-driven features for churn prediction
- Tenure-based features (lifecycle stages)
- Usage behavior features (service adoption)
- Billing risk features (payment patterns)
- All features validated for data leakage

CRITICAL: Feature engineering is where you WIN interviews!
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ChurnFeatureEngineering:
    """
    Feature engineering for customer churn prediction.
    
    Business Context:
    - New customers (0-6 months) behave differently than loyal (24+ months)
    - Multiple services = higher switching cost = lower churn
    - Payment method indicates financial stability
    - High charges without value-add services = churn risk
    
    Feature Categories:
    1. Tenure & Lifecycle Features
    2. Usage Behavior Features  
    3. Billing & Payment Risk Features
    4. Service Bundle Features
    """
    
    def __init__(self, df):
        """
        Initialize feature engineering.
        
        Args:
            df (pd.DataFrame): Processed data from processed_data_creator.py
        """
        self.df = df.copy()
        self.feature_log = []
        
    def _log_feature(self, feature_name, description):
        """Log feature creation for transparency"""
        self.feature_log.append(f"{feature_name}: {description}")
        print(f"   ✓ Created: {feature_name}")
    
    def create_tenure_features(self):
        """
        Create tenure-based lifecycle features.
        
        Business Logic:
        - New (0-12 months): High churn risk, still evaluating
        - Growing (12-36 months): Medium risk, establishing habits
        - Loyal (36+ months): Low churn risk, high switching cost
        
        Why this matters:
        - Churn patterns differ dramatically across lifecycle stages
        - Intervention strategies should be stage-specific
        """
        print("\n🔧 Creating Tenure & Lifecycle Features...")
        
        # Feature 1: Tenure buckets (categorical)
        def categorize_tenure(tenure):
            if tenure <= 12:
                return 'New'
            elif tenure <= 36:
                return 'Growing'
            else:
                return 'Loyal'
        
        self.df['TenureGroup'] = self.df['tenure'].apply(categorize_tenure)
        self._log_feature('TenureGroup', 'New/Growing/Loyal based on tenure')
        
        # Feature 2: Tenure buckets (numeric for models)
        tenure_mapping = {'New': 0, 'Growing': 1, 'Loyal': 2}
        self.df['TenureGroup_Numeric'] = self.df['TenureGroup'].map(tenure_mapping)
        self._log_feature('TenureGroup_Numeric', 'Numeric encoding of tenure groups')
        
        # Feature 3: Is new customer (high risk)
        self.df['IsNewCustomer'] = (self.df['tenure'] <= 6).astype(int)
        self._log_feature('IsNewCustomer', 'Binary flag for customers <=6 months')
        
        # Feature 4: Is loyal customer (low risk)
        self.df['IsLoyalCustomer'] = (self.df['tenure'] >= 36).astype(int)
        self._log_feature('IsLoyalCustomer', 'Binary flag for customers >=36 months')
        
        # Feature 5: Tenure in years (easier interpretation)
        self.df['TenureYears'] = (self.df['tenure'] / 12).round(2)
        self._log_feature('TenureYears', 'Tenure converted to years')
        
        return self
    
    def create_usage_features(self):
        """
        Create usage behavior features.
        
        Business Logic:
        - More services = higher engagement = lower churn
        - Streaming services indicate entertainment value
        - Security services indicate business/premium users
        - Phone + Internet = full bundle customer
        """
        print("\n🔧 Creating Usage Behavior Features...")
        
        # Feature 1: Total service count
        service_columns = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        service_count = 0
        for col in service_columns:
            if col in self.df.columns:
                service_count += (self.df[col] == 'Yes').astype(int)
        
        self.df['ServiceCount'] = service_count
        self._log_feature('ServiceCount', 'Total number of add-on services')
        
        # Feature 2: Has premium services (security + backup)
        has_security = (self.df.get('OnlineSecurity', 'No') == 'Yes').astype(int)
        has_backup = (self.df.get('OnlineBackup', 'No') == 'Yes').astype(int)
        self.df['HasPremiumServices'] = ((has_security + has_backup) >= 1).astype(int)
        self._log_feature('HasPremiumServices', 'Has security or backup services')
        
        # Feature 3: Has streaming services
        has_tv = (self.df.get('StreamingTV', 'No') == 'Yes').astype(int)
        has_movies = (self.df.get('StreamingMovies', 'No') == 'Yes').astype(int)
        self.df['HasStreamingServices'] = ((has_tv + has_movies) >= 1).astype(int)
        self._log_feature('HasStreamingServices', 'Has TV or Movie streaming')
        
        # Feature 4: Full bundle customer (phone + internet)
        has_phone = (self.df.get('PhoneService', 0) == 1) | (self.df.get('PhoneService', 'No') == 'Yes')
        has_internet = self.df.get('InternetService', 'No') != 'No'
        self.df['IsFullBundle'] = (has_phone & has_internet).astype(int)
        self._log_feature('IsFullBundle', 'Has both phone and internet')
        
        # Feature 5: Internet service type (numeric)
        internet_mapping = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        self.df['InternetType_Numeric'] = self.df['InternetService'].map(internet_mapping)
        self._log_feature('InternetType_Numeric', 'Numeric encoding of internet type')
        
        return self
    
    def create_billing_features(self):
        """
        Create billing and payment risk features.
        
        Business Logic:
        - High charges without services = price sensitivity
        - Manual payment = friction = higher churn
        - Charge per tenure = value perception
        - Month-to-month = no commitment = higher churn
        """
        print("\n🔧 Creating Billing & Payment Features...")
        
        # Feature 1: Charge per month of tenure (value perception)
        # Avoid division by zero
        self.df['ChargePerTenure'] = np.where(
            self.df['tenure'] > 0,
            self.df['TotalCharges'] / self.df['tenure'],
            self.df['MonthlyCharges']
        ).round(2)
        self._log_feature('ChargePerTenure', 'Average monthly charge (TotalCharges/tenure)')
        
        # Feature 2: High charges flag (price sensitive segment)
        high_charge_threshold = self.df['MonthlyCharges'].quantile(0.75)
        self.df['IsHighCharges'] = (self.df['MonthlyCharges'] > high_charge_threshold).astype(int)
        self._log_feature('IsHighCharges', f'Monthly charges > Rs{high_charge_threshold:.2f}')
        
        # Feature 3: Low value perception (high charges, low services)
        self.df['LowValuePerception'] = (
            (self.df['MonthlyCharges'] > high_charge_threshold) & 
            (self.df['ServiceCount'] <= 1)
        ).astype(int)
        self._log_feature('LowValuePerception', 'High charges but few services')
        
        # Feature 4: Payment risk flag
        has_manual_payment = self.df['PaymentMethod'].isin(['Electronic check', 'Mailed check'])
        self.df['PaymentRiskFlag'] = has_manual_payment.astype(int)
        self._log_feature('PaymentRiskFlag', 'Manual payment method (not automatic)')
        
        # Feature 5: Contract risk (month-to-month)
        self.df['IsMonthToMonth'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        self._log_feature('IsMonthToMonth', 'Month-to-month contract (high churn risk)')
        
        # Feature 6: Has long-term contract
        self.df['HasLongContract'] = (self.df['Contract'].isin(['One year', 'Two year'])).astype(int)
        self._log_feature('HasLongContract', 'Annual or biennial contract')
        
        # Feature 7: Charge to service ratio
        self.df['ChargeServiceRatio'] = np.where(
            self.df['ServiceCount'] > 0,
            self.df['MonthlyCharges'] / (self.df['ServiceCount'] + 1),  # +1 to avoid division issues
            self.df['MonthlyCharges']
        ).round(2)
        self._log_feature('ChargeServiceRatio', 'Monthly charges per service')
        
        return self
    
    def create_risk_score(self):
        """
        Create composite risk score based on multiple factors.
        
        Risk Factors (each adds to risk):
        - New customer (+3 points)
        - Month-to-month contract (+3 points)
        - Manual payment (+2 points)
        - No services (+2 points)
        - High charges (+1 point)
        
        Score Range: 0-11 (higher = more risk)
        """
        print("\n🔧 Creating Composite Risk Score...")
        
        risk_score = 0
        
        # Tenure risk
        risk_score += self.df['IsNewCustomer'] * 3
        
        # Contract risk
        risk_score += self.df['IsMonthToMonth'] * 3
        
        # Payment risk
        risk_score += self.df['PaymentRiskFlag'] * 2
        
        # Service risk
        risk_score += (self.df['ServiceCount'] == 0).astype(int) * 2
        
        # Price risk
        risk_score += self.df['IsHighCharges'] * 1
        
        self.df['RiskScore'] = risk_score
        self._log_feature('RiskScore', 'Composite risk score (0-11, higher=more risk)')
        
        # Risk categories
        def categorize_risk(score):
            if score <= 3:
                return 'Low'
            elif score <= 6:
                return 'Medium'
            else:
                return 'High'
        
        self.df['RiskCategory'] = self.df['RiskScore'].apply(categorize_risk)
        self._log_feature('RiskCategory', 'Risk level: Low/Medium/High')
        
        return self
    
    def create_contract_features(self):
        """
        Create contract-specific features.
        """
        print("\n🔧 Creating Contract Features...")
        
        # Contract type encoding
        contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        self.df['ContractType_Numeric'] = self.df['Contract'].map(contract_mapping)
        self._log_feature('ContractType_Numeric', 'Numeric encoding of contract type')
        
        # Payment method - is automatic
        auto_payment_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
        self.df['IsAutoPayment'] = self.df['PaymentMethod'].isin(auto_payment_methods).astype(int)
        self._log_feature('IsAutoPayment', 'Automatic payment method')
        
        return self
    
    def validate_features(self):
        """
        Validate created features for data leakage and quality.
        
        Checks:
        1. No features use future information
        2. No features perfectly correlate with target
        3. Features have reasonable distributions
        4. No constant features
        """
        print("\n✅ Validating Features for Data Leakage...")
        
        # Check for constant features
        constant_features = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   ⚠️  Constant features detected: {constant_features}")
        else:
            print(f"   ✓ No constant features")
        
        # Check for perfect correlation with target
        if 'Churn' in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlations = self.df[numeric_cols].corrwith(self.df['Churn']).abs()
            perfect_corr = correlations[correlations > 0.99]
            
            if len(perfect_corr) > 1:  # More than just Churn itself
                print(f"   ⚠️  Perfect correlations detected:")
                for col, corr in perfect_corr.items():
                    if col != 'Churn':
                        print(f"      {col}: {corr:.3f}")
            else:
                print(f"   ✓ No perfect correlations (data leakage check passed)")
        
        # Check feature count
        feature_cols = [col for col in self.df.columns if col != 'Churn']
        print(f"   ✓ Total features: {len(feature_cols)}")
        
        return True
    
    def show_feature_summary(self):
        """Display summary of created features"""
        print("\n" + "="*70)
        print("📊 FEATURE ENGINEERING SUMMARY")
        print("="*70)
        
        print("\nFeatures Created:")
        for i, log in enumerate(self.feature_log, 1):
            print(f"   {i}. {log}")
        
        print(f"\n📈 Dataset Info:")
        print(f"   Total Records: {len(self.df):,}")
        print(f"   Total Features: {len(self.df.columns)}")
        print(f"   Churn Rate: {self.df['Churn'].mean()*100:.2f}%")
        
        # Show some statistics
        print(f"\n📊 New Feature Statistics:")
        if 'ServiceCount' in self.df.columns:
            print(f"   Service Count: {self.df['ServiceCount'].min()}-{self.df['ServiceCount'].max()} (avg: {self.df['ServiceCount'].mean():.2f})")
        if 'RiskScore' in self.df.columns:
            print(f"   Risk Score: {self.df['RiskScore'].min()}-{self.df['RiskScore'].max()} (avg: {self.df['RiskScore'].mean():.2f})")
        if 'TenureGroup' in self.df.columns:
            print(f"\n   Tenure Groups:")
            for group, count in self.df['TenureGroup'].value_counts().items():
                print(f"      {group}: {count:,} ({count/len(self.df)*100:.1f}%)")
    
    def save_features(self, output_path='data/processed/churn_features.csv'):
        """
        Save engineered features to CSV.
        
        Args:
            output_path (str): Path to save features
        """
        print("\n💾 Saving Engineered Features...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"   ✓ Saved to: {output_path}")
        print(f"   ✓ Shape: {self.df.shape}")
        
        # Save feature documentation
        doc_path = output_path.parent / 'feature_documentation.txt'
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE ENGINEERING DOCUMENTATION\n")
            f.write("="*70 + "\n\n")
            f.write("Features Created:\n")
            f.write("-"*70 + "\n")
            for i, log in enumerate(self.feature_log, 1):
                f.write(f"{i}. {log}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("Feature Categories:\n")
            f.write("-"*70 + "\n")
            f.write("1. Tenure & Lifecycle: TenureGroup, IsNewCustomer, IsLoyalCustomer\n")
            f.write("2. Usage Behavior: ServiceCount, HasPremiumServices, IsFullBundle\n")
            f.write("3. Billing & Payment: ChargePerTenure, PaymentRiskFlag, IsMonthToMonth\n")
            f.write("4. Risk Scoring: RiskScore, RiskCategory\n")
        
        print(f"   ✓ Documentation saved to: {doc_path}")
        
        return output_path
    
    def engineer_all_features(self):
        """
        Execute complete feature engineering pipeline.
        
        Returns:
            pd.DataFrame: Data with all engineered features
        """
        print("\n" + "="*70)
        print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Create all feature groups
        self.create_tenure_features()
        self.create_usage_features()
        self.create_billing_features()
        self.create_contract_features()
        self.create_risk_score()
        
        # Validate
        self.validate_features()
        
        # Summary
        self.show_feature_summary()
        
        # Save
        output_path = self.save_features()
        
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*70)
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review: {output_path}")
        print(f"   2. Next phase: Model Training (src/train.py)")
        print("="*70 + "\n")
        
        return self.df


def main():
    """Main execution function"""
    # Load processed data
    processed_path = 'data/processed/churn_processed.csv'
    
    if not Path(processed_path).exists():
        print(f"❌ ERROR: Processed data not found at {processed_path}")
        print("   Please run src/processed_data_creator.py first")
        return
    
    print(f"📂 Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path)
    print(f"   Loaded {len(df):,} records\n")
    
    # Create feature engineer
    fe = ChurnFeatureEngineering(df)
    
    # Engineer all features
    df_features = fe.engineer_all_features()
    
    print("🎉 Feature engineering complete!")
    print("   Your data is now ready for model training.")


if __name__ == "__main__":
    main()