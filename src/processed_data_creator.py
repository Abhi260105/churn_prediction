"""
Processed Data Creator for Churn Prediction Project
File: src/processed_data_creator.py

Purpose: Convert raw data into clean, analysis-ready format
FIXED: Unicode encoding issues for Windows
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

class ProcessedDataCreator:
    """
    Processes raw churn data into clean, analysis-ready format.
    
    Key Responsibilities:
    1. Remove data leakage risks (customerID, future info)
    2. Handle missing values with business logic
    3. Fix data types and encodings
    4. Basic data quality checks
    5. Prepare for feature engineering
    """
    
    def __init__(self, raw_data_path):
        """Initialize the data processor."""
        self.raw_data_path = Path(raw_data_path)
        self.df = None
        self.processing_steps = []
        
    def _log_step(self, step_name, details):
        """Log a processing step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {step_name}: {details}"
        self.processing_steps.append(log_entry)
        print(f"   ✓ {details}")
    
    def load_raw_data(self):
        """Load raw data from CSV."""
        print("="*70)
        print("📂 LOADING RAW DATA FOR PROCESSING")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.raw_data_path)
            print(f"✅ Loaded: {self.raw_data_path}")
            print(f"   Shape: {self.df.shape}")
            self._log_step("Load", f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            sys.exit(1)
    
    def remove_data_leakage(self):
        """Remove features that would cause data leakage."""
        print("\n" + "="*70)
        print("🔒 REMOVING DATA LEAKAGE FEATURES")
        print("="*70)
        
        original_cols = len(self.df.columns)
        
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
            self._log_step(
                "Remove Leakage", 
                "Dropped 'customerID' (identifier only, no predictive value)"
            )
        
        new_cols = len(self.df.columns)
        print(f"\n📊 Columns: {original_cols} → {new_cols}")
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values using business logic."""
        print("\n" + "="*70)
        print("🔧 HANDLING MISSING VALUES")
        print("="*70)
        
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before processing: {missing_before}")
        
        if 'TotalCharges' in self.df.columns:
            # Convert to numeric
            self.df['TotalCharges'] = pd.to_numeric(
                self.df['TotalCharges'], 
                errors='coerce'
            )
            
            missing_total_charges = self.df['TotalCharges'].isnull().sum()
            print(f"\nTotalCharges missing: {missing_total_charges}")
            
            if missing_total_charges > 0:
                # Strategy 1: New customers (tenure = 0)
                new_customer_mask = (self.df['TotalCharges'].isnull()) & (self.df['tenure'] == 0)
                new_customers_count = new_customer_mask.sum()
                
                if new_customers_count > 0:
                    self.df.loc[new_customer_mask, 'TotalCharges'] = 0
                    self._log_step(
                        "Missing Values",
                        f"Filled {new_customers_count} TotalCharges for new customers (tenure=0) with 0"
                    )
                
                # Strategy 2: Existing customers - use median
                remaining_nulls = self.df['TotalCharges'].isnull().sum()
                
                if remaining_nulls > 0:
                    median_value = self.df['TotalCharges'].median()
                    # FIXED: Use proper assignment instead of inplace
                    self.df['TotalCharges'] = self.df['TotalCharges'].fillna(median_value)
                    # Format with Rs instead of Rupee symbol
                    self._log_step(
                        "Missing Values",
                        f"Filled {remaining_nulls} TotalCharges for existing customers with median: Rs{median_value:.2f}"
                    )
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\nMissing values after processing: {missing_after}")
        self._log_step("Missing Values", f"Reduced missing values: {missing_before} → {missing_after}")
        
        return self
    
    def fix_data_types(self):
        """Fix data types for all columns."""
        print("\n" + "="*70)
        print("📊 FIXING DATA TYPES")
        print("="*70)
        
        # Convert target variable to binary
        if 'Churn' in self.df.columns:
            original_values = self.df['Churn'].unique()
            self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
            self._log_step(
                "Data Types",
                f"Converted Churn: {list(original_values)} → [0, 1]"
            )
        
        # Convert Yes/No columns to binary
        yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        converted_count = 0
        
        for col in yes_no_columns:
            if col in self.df.columns:
                self.df[col] = (self.df[col] == 'Yes').astype(int)
                converted_count += 1
        
        if converted_count > 0:
            self._log_step(
                "Data Types",
                f"Converted {converted_count} Yes/No columns to binary (0/1): {', '.join(yes_no_columns)}"
            )
        
        # Ensure numeric columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self._log_step(
            "Data Types",
            f"Ensured numeric type for: {', '.join(numeric_columns)}"
        )
        
        print("\n📋 Final Data Types:")
        print(self.df.dtypes.value_counts())
        
        return self
    
    def validate_processed_data(self):
        """Validate the processed data before saving."""
        print("\n" + "="*70)
        print("✅ VALIDATING PROCESSED DATA")
        print("="*70)
        
        validation_passed = True
        
        # Check 1: Critical columns have no missing values
        critical_columns = ['tenure', 'MonthlyCharges', 'Churn']
        for col in critical_columns:
            if col in self.df.columns:
                missing = self.df[col].isnull().sum()
                if missing > 0:
                    print(f"   ❌ {col} has {missing} missing values")
                    validation_passed = False
                else:
                    print(f"   ✓ {col} has no missing values")
        
        # Check 2: Churn is binary
        if 'Churn' in self.df.columns:
            unique_churn = self.df['Churn'].unique()
            if set(unique_churn).issubset({0, 1}):
                print(f"   ✓ Churn is binary: {unique_churn}")
            else:
                print(f"   ❌ Churn is not binary: {unique_churn}")
                validation_passed = False
        
        # Check 3: Tenure is non-negative
        if 'tenure' in self.df.columns:
            if (self.df['tenure'] >= 0).all():
                print(f"   ✓ Tenure is non-negative (min: {self.df['tenure'].min()})")
            else:
                print(f"   ❌ Tenure has negative values")
                validation_passed = False
        
        # Check 4: Charges are positive
        if 'MonthlyCharges' in self.df.columns:
            if (self.df['MonthlyCharges'] > 0).all():
                print(f"   ✓ MonthlyCharges are positive (min: Rs{self.df['MonthlyCharges'].min():.2f})")
            else:
                print(f"   ❌ MonthlyCharges has non-positive values")
                validation_passed = False
        
        # Check 5: Dataset size
        print(f"   ✓ Final dataset size: {len(self.df):,} records")
        
        if validation_passed:
            print("\n✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("\n❌ SOME VALIDATION CHECKS FAILED")
        
        return validation_passed
    
    def show_processing_summary(self):
        """Display summary of all processing steps."""
        print("\n" + "="*70)
        print("📋 PROCESSING SUMMARY")
        print("="*70)
        
        for i, step in enumerate(self.processing_steps, 1):
            print(f"{i}. {step}")
        
        print("\n" + "="*70)
    
    def save_processed_data(self, output_path='data/processed/churn_processed.csv'):
        """Save processed data to CSV."""
        print("\n" + "="*70)
        print("💾 SAVING PROCESSED DATA")
        print("="*70)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to: {output_path}")
        print(f"   Shape: {self.df.shape}")
        print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")
        
        # Save processing log - FIXED: Use UTF-8 encoding
        log_path = output_path.parent / 'processing_log.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DATA PROCESSING LOG\n")
            f.write("="*70 + "\n\n")
            f.write(f"Input File: {self.raw_data_path}\n")
            f.write(f"Output File: {output_path}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("PROCESSING STEPS:\n")
            f.write("-"*70 + "\n")
            for i, step in enumerate(self.processing_steps, 1):
                f.write(f"{i}. {step}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("FINAL STATISTICS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Features: {len(self.df.columns)}\n")
            f.write(f"Churn Rate: {self.df['Churn'].mean()*100:.2f}%\n")
            f.write(f"Missing Values: {self.df.isnull().sum().sum()}\n")
        
        print(f"📝 Processing log saved to: {log_path}")
    
    def show_final_statistics(self):
        """Display final statistics."""
        print("\n" + "="*70)
        print("📈 FINAL PROCESSED DATA STATISTICS")
        print("="*70)
        
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        if 'Churn' in self.df.columns:
            churn_rate = self.df['Churn'].mean() * 100
            churn_count = self.df['Churn'].sum()
            no_churn_count = len(self.df) - churn_count
            
            print(f"\nChurn Distribution:")
            print(f"   No Churn (0): {no_churn_count:,} ({100-churn_rate:.2f}%)")
            print(f"   Churn (1):    {churn_count:,} ({churn_rate:.2f}%)")
        
        print(f"\nMissing Values: {self.df.isnull().sum().sum()}")
        
        print(f"\nKey Statistics:")
        print(f"   Tenure Range: {self.df['tenure'].min()} - {self.df['tenure'].max()} months")
        print(f"   Monthly Charges: Rs{self.df['MonthlyCharges'].min():.2f} - Rs{self.df['MonthlyCharges'].max():.2f}")
        print(f"   Total Charges: Rs{self.df['TotalCharges'].min():.2f} - Rs{self.df['TotalCharges'].max():.2f}")
    
    def process_pipeline(self):
        """Execute the complete data processing pipeline."""
        print("\n" + "="*70)
        print("🚀 STARTING DATA PROCESSING PIPELINE")
        print("="*70)
        print("\n📖 Business Context:")
        print("   Goal: Predict customers likely to churn in next 30 days")
        print("   Churn: Inactive 60+ days OR subscription cancelled")
        print("   Purpose: Enable proactive retention interventions")
        print("="*70)
        
        self.load_raw_data()
        self.remove_data_leakage()
        self.handle_missing_values()
        self.fix_data_types()
        
        is_valid = self.validate_processed_data()
        
        if not is_valid:
            print("\n❌ Processing failed validation.")
            sys.exit(1)
        
        self.save_processed_data()
        self.show_processing_summary()
        self.show_final_statistics()
        
        print("\n" + "="*70)
        print("✅ DATA PROCESSING COMPLETE")
        print("="*70)
        print("\n🎯 Next Steps:")
        print("   1. Review: data/processed/churn_processed.csv")
        print("   2. Check log: data/processed/processing_log.txt")
        print("   3. Next phase: Feature Engineering (src/features.py)")
        print("="*70 + "\n")
        
        return self.df


def main():
    """Main execution function."""
    raw_data_path = 'data/raw/telco_churn.csv'
    
    if not Path(raw_data_path).exists():
        print(f"❌ ERROR: Raw data file not found at {raw_data_path}")
        print("   Please run data/raw/generate_sample_data.py first.")
        sys.exit(1)
    
    processor = ProcessedDataCreator(raw_data_path)
    processed_df = processor.process_pipeline()
    
    print("🎉 SUCCESS! Your data is now ready for feature engineering.")


if __name__ == "__main__":
    main()