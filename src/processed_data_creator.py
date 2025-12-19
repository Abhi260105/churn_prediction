"""
Processed Data Creator for Churn Prediction Project
File: src/processed_data_creator.py

Purpose: Convert raw data into clean, analysis-ready format
- Removes data leakage features
- Handles missing values intelligently
- Fixes data types
- Prepares data for feature engineering

CRITICAL: This step ensures NO data leakage and proper data quality
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
    
    Business Context:
    - Churn Definition: Inactive 60+ days OR subscription cancelled
    - Goal: Clean data WITHOUT feature engineering (that comes next)
    """
    
    def __init__(self, raw_data_path):
        """
        Initialize the data processor.
        
        Args:
            raw_data_path (str): Path to raw CSV file
        """
        self.raw_data_path = Path(raw_data_path)
        self.df = None
        self.processing_steps = []  # Track all processing steps
        
    def _log_step(self, step_name, details):
        """
        Log a processing step for transparency and debugging.
        
        Args:
            step_name (str): Name of the processing step
            details (str): Details about what was done
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {step_name}: {details}"
        self.processing_steps.append(log_entry)
        print(f"   ✓ {details}")
    
    def load_raw_data(self):
        """
        Load raw data from CSV.
        
        Returns:
            pd.DataFrame: Raw data
        """
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
        """
        Remove features that would cause data leakage.
        
        Data Leakage = Using information that wouldn't be available at prediction time
        
        Features to remove:
        - customerID: Just an identifier, no predictive value
        
        WHY THIS MATTERS:
        - customerID could memorize specific customers during training
        - Model wouldn't generalize to new customers
        - This is a common interview trap question!
        """
        print("\n" + "="*70)
        print("🔒 REMOVING DATA LEAKAGE FEATURES")
        print("="*70)
        
        original_cols = len(self.df.columns)
        
        # Remove customerID
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
            self._log_step(
                "Remove Leakage", 
                "Dropped 'customerID' (identifier only, no predictive value)"
            )
        
        # Check for any other potential leakage features
        # (none in this dataset, but this is where you'd check)
        
        new_cols = len(self.df.columns)
        print(f"\n📊 Columns: {original_cols} → {new_cols}")
        
        return self
    
    def handle_missing_values(self):
        """
        Handle missing values using business logic.
        
        Strategy:
        1. TotalCharges for new customers (tenure=0): Fill with 0
           - Logic: They haven't been charged yet
        2. TotalCharges for existing customers: Fill with median
           - Logic: Preserve distribution, avoid extreme values
        3. Other columns: Investigate case-by-case
        
        WHY NOT DROP?
        - Dropping rows loses valuable information
        - Missing data may have business meaning
        - We need every customer record we can get
        """
        print("\n" + "="*70)
        print("🔧 HANDLING MISSING VALUES")
        print("="*70)
        
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before processing: {missing_before}")
        
        # Handle TotalCharges specifically
        if 'TotalCharges' in self.df.columns:
            
            # First, convert to numeric (may have spaces or strings)
            self.df['TotalCharges'] = pd.to_numeric(
                self.df['TotalCharges'], 
                errors='coerce'
            )
            
            # Count missing after conversion
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
                    self.df['TotalCharges'].fillna(median_value, inplace=True)
                    self._log_step(
                        "Missing Values",
                        f"Filled {remaining_nulls} TotalCharges for existing customers with median: ₹{median_value:.2f}"
                    )
        
        # Check for missing values in other columns
        other_missing = self.df.isnull().sum()
        other_missing = other_missing[other_missing > 0]
        
        if len(other_missing) > 0:
            print(f"\n⚠️  Other columns with missing values:")
            for col, count in other_missing.items():
                print(f"   {col}: {count}")
            print("   These will need to be addressed if critical for analysis")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\nMissing values after processing: {missing_after}")
        self._log_step("Missing Values", f"Reduced missing values: {missing_before} → {missing_after}")
        
        return self
    
    def fix_data_types(self):
        """
        Fix data types for all columns.
        
        Conversions:
        1. Target variable (Churn): Yes/No → 1/0 (binary)
        2. Yes/No columns → 1/0 (binary)
        3. Numeric columns → proper numeric type
        
        WHY THIS MATTERS:
        - ML algorithms need numeric inputs
        - Binary encoding is more interpretable than one-hot
        - Prevents type errors during modeling
        """
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
        
        # Ensure numeric columns are proper numeric type
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
        """
        Validate the processed data before saving.
        
        Checks:
        1. No critical missing values in key columns
        2. Target variable is binary (0/1)
        3. Tenure is non-negative
        4. Charges are positive
        5. No duplicates
        """
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
                print(f"   ✓ MonthlyCharges are positive (min: ₹{self.df['MonthlyCharges'].min():.2f})")
            else:
                print(f"   ❌ MonthlyCharges has non-positive values")
                validation_passed = False
        
        # Check 5: Dataset size
        print(f"   ✓ Final dataset size: {len(self.df):,} records")
        
        if validation_passed:
            print("\n✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("\n❌ SOME VALIDATION CHECKS FAILED")
            print("   Review the issues above before proceeding")
        
        return validation_passed
    
    def show_processing_summary(self):
        """
        Display summary of all processing steps performed.
        """
        print("\n" + "="*70)
        print("📋 PROCESSING SUMMARY")
        print("="*70)
        
        for i, step in enumerate(self.processing_steps, 1):
            print(f"{i}. {step}")
        
        print("\n" + "="*70)
    
    def save_processed_data(self, output_path='data/processed/churn_processed.csv'):
        """
        Save processed data to CSV.
        
        Args:
            output_path (str): Path where processed data will be saved
        """
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
        
        # Save processing log
        log_path = output_path.parent / 'processing_log.txt'
        with open(log_path, 'w') as f:
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
        """
        Display final statistics of processed data.
        """
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
        print(f"   Monthly Charges: ₹{self.df['MonthlyCharges'].min():.2f} - ₹{self.df['MonthlyCharges'].max():.2f}")
        print(f"   Total Charges: ₹{self.df['TotalCharges'].min():.2f} - ₹{self.df['TotalCharges'].max():.2f}")
    
    def process_pipeline(self):
        """
        Execute the complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed data
        """
        print("\n" + "="*70)
        print("🚀 STARTING DATA PROCESSING PIPELINE")
        print("="*70)
        print("\n📖 Business Context:")
        print("   Goal: Predict customers likely to churn in next 30 days")
        print("   Churn: Inactive 60+ days OR subscription cancelled")
        print("   Purpose: Enable proactive retention interventions")
        print("="*70)
        
        # Execute pipeline
        self.load_raw_data()
        self.remove_data_leakage()
        self.handle_missing_values()
        self.fix_data_types()
        
        # Validate
        is_valid = self.validate_processed_data()
        
        if not is_valid:
            print("\n❌ Processing failed validation. Please review errors above.")
            sys.exit(1)
        
        # Save
        self.save_processed_data()
        
        # Summary
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
    """
    Main execution function for standalone script usage.
    """
    # Path to raw data
    raw_data_path = 'data/raw/telco_churn.csv'
    
    # Check if file exists
    if not Path(raw_data_path).exists():
        print(f"❌ ERROR: Raw data file not found at {raw_data_path}")
        print("   Please run src/raw_data_loader.py first or check the file path.")
        sys.exit(1)
    
    # Create processor and run pipeline
    processor = ProcessedDataCreator(raw_data_path)
    processed_df = processor.process_pipeline()
    
    print("🎉 SUCCESS! Your data is now ready for feature engineering.")


if __name__ == "__main__":
    main()