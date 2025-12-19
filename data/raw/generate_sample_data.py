"""
Sample Telco Churn Data Generator
File: data/raw/generate_sample_data.py

Purpose: Generate realistic telecom customer churn dataset
- Creates 7,000+ customer records
- Realistic churn patterns based on industry behavior
- Includes all required features for analysis

Run this FIRST to create the telco_churn.csv file
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_telco_churn_data(n_samples=7043, random_state=42):
    """
    Generate realistic telecom customer churn dataset.
    
    Business Scenario:
    - Telecom company with phone and internet services
    - Multiple service add-ons (streaming, security, etc.)
    - Different contract types and payment methods
    - Goal: Predict which customers will churn
    
    Args:
        n_samples (int): Number of customer records to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated dataset with realistic churn patterns
    """
    
    print("="*70)
    print("🎲 GENERATING SAMPLE TELCO CHURN DATA")
    print("="*70)
    print(f"\nGenerating {n_samples:,} customer records...")
    
    np.random.seed(random_state)
    
    # ==========================================
    # 1. CUSTOMER IDENTIFIERS
    # ==========================================
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    
    # ==========================================
    # 2. DEMOGRAPHICS
    # ==========================================
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # ==========================================
    # 3. TENURE (Critical for churn prediction)
    # ==========================================
    # Realistic distribution: Many new, some mid-term, fewer long-term
    tenure_new = np.random.exponential(12, int(n_samples * 0.40))      # 0-12 months
    tenure_mid = np.random.uniform(12, 36, int(n_samples * 0.30))      # 1-3 years
    tenure_loyal = np.random.uniform(36, 72, int(n_samples * 0.30))    # 3-6 years
    
    tenure = np.concatenate([tenure_new, tenure_mid, tenure_loyal])
    tenure = np.clip(tenure, 0, 72).astype(int)
    np.random.shuffle(tenure)
    
    # ==========================================
    # 4. PHONE SERVICES
    # ==========================================
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
        'No phone service'
    )
    
    # ==========================================
    # 5. INTERNET SERVICES
    # ==========================================
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        n_samples, 
        p=[0.34, 0.44, 0.22]
    )
    
    # Internet-dependent services (only if customer has internet)
    def internet_dependent_service(service_prob_yes):
        """Helper to create internet-dependent service column"""
        service = np.where(
            internet_service != 'No',
            np.random.choice(['Yes', 'No'], n_samples, p=[service_prob_yes, 1-service_prob_yes]),
            'No internet service'
        )
        return service
    
    online_security = internet_dependent_service(0.29)
    online_backup = internet_dependent_service(0.34)
    device_protection = internet_dependent_service(0.34)
    tech_support = internet_dependent_service(0.29)
    streaming_tv = internet_dependent_service(0.38)
    streaming_movies = internet_dependent_service(0.39)
    
    # ==========================================
    # 6. CONTRACT AND BILLING
    # ==========================================
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        n_samples, 
        p=[0.55, 0.21, 0.24]
    )
    
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    
    payment_method = np.random.choice([
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ], n_samples, p=[0.34, 0.23, 0.22, 0.21])
    
    # ==========================================
    # 7. CHARGES (Realistic pricing model)
    # ==========================================
    
    # Base charge
    base_charge = 20
    monthly_charges = np.full(n_samples, base_charge, dtype=float)
    
    # Add service charges
    monthly_charges += (phone_service == 'Yes') * 15
    monthly_charges += (multiple_lines == 'Yes') * 10
    monthly_charges += (internet_service == 'DSL') * 30
    monthly_charges += (internet_service == 'Fiber optic') * 50
    monthly_charges += (online_security == 'Yes') * 5
    monthly_charges += (online_backup == 'Yes') * 5
    monthly_charges += (device_protection == 'Yes') * 5
    monthly_charges += (tech_support == 'Yes') * 5
    monthly_charges += (streaming_tv == 'Yes') * 10
    monthly_charges += (streaming_movies == 'Yes') * 10
    
    # Add realistic noise
    monthly_charges += np.random.normal(0, 5, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    monthly_charges = np.round(monthly_charges, 2)
    
    # Total charges = monthly × tenure (with some variance)
    total_charges = monthly_charges * tenure * np.random.uniform(0.95, 1.05, n_samples)
    total_charges = np.round(total_charges, 2)
    total_charges[tenure == 0] = 0  # New customers haven't been charged
    
    # ==========================================
    # 8. CHURN (TARGET VARIABLE - Realistic Logic)
    # ==========================================
    
    print("\nApplying realistic churn logic...")
    
    # Start with base churn probability
    churn_probability = np.full(n_samples, 0.15)
    
    # Factor 1: Tenure (new customers churn more)
    churn_probability += np.where(tenure < 6, 0.25, 0)
    churn_probability += np.where(tenure < 12, 0.15, 0)
    churn_probability -= np.where(tenure > 24, 0.15, 0)
    churn_probability -= np.where(tenure > 48, 0.10, 0)
    
    # Factor 2: Contract type (month-to-month highest risk)
    churn_probability += np.where(contract == 'Month-to-month', 0.25, 0)
    churn_probability -= np.where(contract == 'Two year', 0.20, 0)
    
    # Factor 3: Payment method (automatic = lower churn)
    churn_probability += np.where(payment_method == 'Electronic check', 0.15, 0)
    churn_probability -= np.where(
        payment_method.astype(str).str.contains('automatic', case=False),
        0.10,
        0
    )
    
    # Factor 4: Service bundle (more services = lower churn)
    service_count = (
        (online_security == 'Yes').astype(int) +
        (online_backup == 'Yes').astype(int) +
        (device_protection == 'Yes').astype(int) +
        (tech_support == 'Yes').astype(int)
    )
    churn_probability -= service_count * 0.03
    
    # Factor 5: Price sensitivity (high charges increase churn)
    churn_probability += np.where(monthly_charges > 80, 0.12, 0)
    
    # Factor 6: Internet service type (fiber = higher churn due to price)
    churn_probability += np.where(internet_service == 'Fiber optic', 0.08, 0)
    
    # Factor 7: Senior citizens (slightly higher churn)
    churn_probability += senior_citizen * 0.05
    
    # Clip probability to reasonable range
    churn_probability = np.clip(churn_probability, 0.05, 0.80)
    
    # Generate actual churn based on probability
    churn_binary = (np.random.random(n_samples) < churn_probability).astype(int)
    churn = np.where(churn_binary == 1, 'Yes', 'No')
    
    # ==========================================
    # 9. INTRODUCE REALISTIC MISSING VALUES
    # ==========================================
    
    # TotalCharges sometimes missing for new customers (data quality issue)
    total_charges_with_missing = total_charges.copy()
    missing_mask = np.random.choice(n_samples, size=int(n_samples * 0.002), replace=False)
    total_charges_with_missing = total_charges_with_missing.astype(object)
    total_charges_with_missing[missing_mask] = ' '  # Empty string (mimics real data)
    
    # ==========================================
    # 10. CREATE DATAFRAME
    # ==========================================
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_with_missing,
        'Churn': churn
    })
    
    # ==========================================
    # 11. SHOW GENERATION STATISTICS
    # ==========================================
    
    churn_rate = (churn == 'Yes').sum() / n_samples * 100
    
    print("\n✅ Data Generation Complete!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Churn Rate: {churn_rate:.2f}%")
    print(f"   No Churn: {(churn == 'No').sum():,} ({100-churn_rate:.2f}%)")
    print(f"   Churn: {(churn == 'Yes').sum():,} ({churn_rate:.2f}%)")
    print(f"   Missing Values: {(total_charges_with_missing == ' ').sum()}")
    
    print(f"\n📈 Tenure Distribution:")
    print(f"   New (0-12 months): {(tenure <= 12).sum():,} ({(tenure <= 12).sum()/n_samples*100:.1f}%)")
    print(f"   Growing (12-36 months): {((tenure > 12) & (tenure <= 36)).sum():,} ({((tenure > 12) & (tenure <= 36)).sum()/n_samples*100:.1f}%)")
    print(f"   Loyal (36+ months): {(tenure > 36).sum():,} ({(tenure > 36).sum()/n_samples*100:.1f}%)")
    
    print(f"\n💰 Charges Statistics:")
    print(f"   Monthly Charges: ₹{monthly_charges.min():.2f} - ₹{monthly_charges.max():.2f}")
    print(f"   Average Monthly: ₹{monthly_charges.mean():.2f}")
    print(f"   Total Charges: ₹{total_charges.min():.2f} - ₹{total_charges.max():.2f}")
    
    return df


def main():
    """
    Main function to generate and save data.
    """
    print("\n" + "="*70)
    print("🚀 TELCO CHURN DATA GENERATION")
    print("="*70 + "\n")
    
    # Generate data
    df = generate_telco_churn_data(n_samples=7043, random_state=42)
    
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / 'telco_churn.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Show sample
    print("\n" + "="*70)
    print("📋 SAMPLE DATA (First 3 rows)")
    print("="*70)
    print(df.head(3).to_string())
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE")
    print("="*70)
    print("\n🎯 Next Steps:")
    print("   1. Run: python src/raw_data_loader.py")
    print("   2. Then: python src/processed_data_creator.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()