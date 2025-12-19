"""
Raw Data Loader for Churn Prediction Project
File: src/raw_data_loader.py

Purpose: Load and validate raw telco churn data
- Performs basic data quality checks
- Validates required columns exist
- Shows churn distribution
- Detects duplicates and missing values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

class RawDataLoader:
    """
    Loads and validates raw customer churn data from CSV.
    
    Business Context:
    - Goal: Predict customers likely to churn in next 30 days
    - Churn Definition: Customer inactive 60+ days OR subscription cancelled
    - Dataset: Telco customer subscription and usage data
    """
    
    def __init__(self, data_path):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to raw CSV file (e.g., 'data/raw/telco_churn.csv')
        """
        self.data_path = Path(data_path)
        self.df = None
        self.validation_results = {}
        
    def load_data(self):
        """
        Load raw CSV data into pandas DataFrame.
        
        Returns:
            pd.DataFrame: Raw data
        """
        print("="*70)
        print("📂 LOADING RAW DATA")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Successfully loaded: {self.data_path}")
            print(f"   Records: {len(self.df):,}")
            print(f"   Columns: {len(self.df.columns)}")
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ ERROR: File not found at {self.data_path}")
            print("   Please ensure the file exists at the specified path.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            sys.exit(1)
    
    def validate_required_columns(self):
        """
        Check if all required columns are present in the dataset.
        
        Required columns for churn prediction:
        - customerID: Unique identifier
        - tenure: Months with company
        - MonthlyCharges: Monthly billing amount
        - TotalCharges: Total amount billed
        - Churn: Target variable (Yes/No)
        """
        print("\n" + "="*70)
        print("🔍 VALIDATING REQUIRED COLUMNS")
        print("="*70)
        
        required_columns = {
            'customerID': 'Unique customer identifier',
            'tenure': 'Months customer has been with company',
            'MonthlyCharges': 'Monthly billing amount',
            'TotalCharges': 'Total amount billed to date',
            'Churn': 'Target variable (Yes/No)'
        }
        
        missing_columns = []
        present_columns = []
        
        for col, description in required_columns.items():
            if col in self.df.columns:
                present_columns.append(col)
                print(f"   ✅ {col}: {description}")
            else:
                missing_columns.append(col)
                print(f"   ❌ {col}: MISSING - {description}")
        
        self.validation_results['required_columns'] = {
            'present': present_columns,
            'missing': missing_columns
        }
        
        if missing_columns:
            print(f"\n❌ VALIDATION FAILED: Missing {len(missing_columns)} required columns")
            print("   Cannot proceed with missing required columns.")
            return False
        else:
            print(f"\n✅ All {len(required_columns)} required columns present")
            return True
    
    def check_churn_distribution(self):
        """
        Analyze the distribution of the target variable (Churn).
        
        Critical for understanding class imbalance.
        Typical churn rate: 15-30% in telecom industry.
        """
        print("\n" + "="*70)
        print("📊 CHURN DISTRIBUTION ANALYSIS")
        print("="*70)
        
        if 'Churn' not in self.df.columns:
            print("❌ Churn column not found. Cannot analyze distribution.")
            return
        
        # Get value counts
        churn_counts = self.df['Churn'].value_counts()
        churn_pct = self.df['Churn'].value_counts(normalize=True) * 100
        
        print("\nAbsolute Counts:")
        for value, count in churn_counts.items():
            print(f"   {value}: {count:,} customers")
        
        print("\nPercentage Distribution:")
        for value, pct in churn_pct.items():
            bar_length = int(pct / 2)  # Scale for visualization
            bar = "█" * bar_length
            print(f"   {value}: {pct:5.2f}% {bar}")
        
        # Calculate churn rate
        if 'Yes' in churn_pct.index:
            churn_rate = churn_pct['Yes']
        else:
            churn_rate = churn_pct.get(1, 0)  # Handle binary encoding
        
        print(f"\n🎯 Churn Rate: {churn_rate:.2f}%")
        
        # Provide context
        if churn_rate < 10:
            print("   ⚠️  WARNING: Very low churn rate (<10%)")
            print("   Consider: Severe class imbalance - will need special handling")
        elif churn_rate < 20:
            print("   ℹ️  Low churn rate (10-20%)")
            print("   Consider: Class imbalance handling (SMOTE or class weights)")
        elif churn_rate < 35:
            print("   ✅ Moderate churn rate (20-35%) - typical for telecom")
            print("   Consider: Standard classification approaches will work")
        else:
            print("   ⚠️  High churn rate (>35%)")
            print("   Consider: Investigate business issues causing high churn")
        
        self.validation_results['churn_rate'] = churn_rate
        
    def check_duplicates(self):
        """
        Check for duplicate customer records.
        
        Duplicates can indicate:
        - Data quality issues
        - Multiple records for same customer
        - Need for deduplication logic
        """
        print("\n" + "="*70)
        print("🔍 CHECKING FOR DUPLICATES")
        print("="*70)
        
        if 'customerID' not in self.df.columns:
            print("⚠️  customerID column not found. Cannot check for duplicates.")
            return
        
        # Check for duplicate customerIDs
        duplicate_ids = self.df['customerID'].duplicated().sum()
        
        if duplicate_ids > 0:
            print(f"❌ Found {duplicate_ids:,} duplicate customer IDs")
            print("\n   Sample duplicate records:")
            duplicated_customers = self.df[self.df['customerID'].duplicated(keep=False)]
            print(duplicated_customers.head(10))
            print("\n⚠️  ACTION REQUIRED: Investigate and handle duplicates before processing")
        else:
            print(f"✅ No duplicate customer IDs found")
            print(f"   All {len(self.df):,} records have unique customer IDs")
        
        self.validation_results['duplicates'] = duplicate_ids
    
    def check_missing_values(self):
        """
        Detect and report missing values across all columns.
        
        Missing values need special handling:
        - Some may be business logic (e.g., new customers)
        - Others may indicate data quality issues
        """
        print("\n" + "="*70)
        print("🔍 MISSING VALUES ANALYSIS")
        print("="*70)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        
        # Filter only columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing_df) == 0:
            print("✅ No missing values detected in any column")
            print("   Data is complete and ready for processing")
        else:
            print(f"⚠️  Found missing values in {len(missing_df)} columns:\n")
            for _, row in missing_df.iterrows():
                print(f"   {row['Column']:20s}: {row['Missing_Count']:5.0f} "
                      f"({row['Missing_Percentage']:5.2f}%)")
            
            print(f"\n   Total missing values: {missing_df['Missing_Count'].sum():,.0f}")
            print("   ACTION: These will be handled in data processing step")
        
        self.validation_results['missing_values'] = missing_df.to_dict('records')
    
    def show_data_sample(self, n=5):
        """
        Display sample records from the dataset.
        
        Args:
            n (int): Number of sample records to display
        """
        print("\n" + "="*70)
        print(f"📋 SAMPLE DATA (First {n} rows)")
        print("="*70 + "\n")
        
        print(self.df.head(n).to_string())
    
    def show_data_info(self):
        """
        Display detailed information about the dataset.
        
        Includes:
        - Data types of each column
        - Basic statistics for numerical columns
        - Memory usage
        """
        print("\n" + "="*70)
        print("ℹ️  DETAILED DATA INFORMATION")
        print("="*70)
        
        print("\n1. DATASET SHAPE:")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")
        
        print("\n2. COLUMN DATA TYPES:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        print("\n3. MEMORY USAGE:")
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"   Total: {memory_mb:.2f} MB")
        
        print("\n4. NUMERICAL COLUMNS STATISTICS:")
        print(self.df.describe().to_string())
    
    def save_validation_report(self, output_path='reports/raw_data_validation.txt'):
        """
        Save validation results to a text file.
        
        Args:
            output_path (str): Path where report will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RAW DATA VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"File: {self.data_path}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Columns: {len(self.df.columns)}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 70 + "\n")
            
            # Required columns
            f.write("\n1. Required Columns Check:\n")
            if 'required_columns' in self.validation_results:
                present = self.validation_results['required_columns']['present']
                missing = self.validation_results['required_columns']['missing']
                f.write(f"   Present: {len(present)}\n")
                f.write(f"   Missing: {len(missing)}\n")
                if missing:
                    f.write(f"   Missing columns: {', '.join(missing)}\n")
            
            # Churn rate
            f.write("\n2. Churn Distribution:\n")
            if 'churn_rate' in self.validation_results:
                f.write(f"   Churn Rate: {self.validation_results['churn_rate']:.2f}%\n")
            
            # Duplicates
            f.write("\n3. Duplicate Records:\n")
            if 'duplicates' in self.validation_results:
                f.write(f"   Duplicates Found: {self.validation_results['duplicates']}\n")
            
            # Missing values
            f.write("\n4. Missing Values:\n")
            if 'missing_values' in self.validation_results:
                if len(self.validation_results['missing_values']) == 0:
                    f.write("   No missing values detected\n")
                else:
                    for item in self.validation_results['missing_values']:
                        f.write(f"   {item['Column']}: {item['Missing_Count']} "
                               f"({item['Missing_Percentage']:.2f}%)\n")
        
        print(f"\n💾 Validation report saved to: {output_path}")
    
    def run_full_validation(self):
        """
        Run complete validation pipeline.
        
        Returns:
            bool: True if all critical validations pass, False otherwise
        """
        # Load data
        self.load_data()
        
        # Run all validations
        columns_valid = self.validate_required_columns()
        self.check_churn_distribution()
        self.check_duplicates()
        self.check_missing_values()
        self.show_data_sample()
        
        # Save report
        self.save_validation_report()
        
        print("\n" + "="*70)
        print("✅ RAW DATA VALIDATION COMPLETE")
        print("="*70)
        
        return columns_valid


def main():
    """
    Main execution function for standalone script usage.
    """
    # Path to raw data
    raw_data_path = 'data/raw/telco_churn.csv'
    
    # Create loader instance
    loader = RawDataLoader(raw_data_path)
    
    # Run full validation
    is_valid = loader.run_full_validation()
    
    if is_valid:
        print("\n🎉 Data is valid and ready for processing!")
        print("   Next step: Run src/processed_data_creator.py")
    else:
        print("\n❌ Data validation failed. Please fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()