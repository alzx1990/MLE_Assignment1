import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, when, log1p, lit, array
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def apply_log_transformations_spark(df):
    log_features = [
        "Annual_Income", 
        "Monthly_Inhand_Salary", 
        "Outstanding_Debt",
        "Total_EMI_per_month", 
        "Amount_invested_monthly", 
        "Num_Credit_Inquiries"
    ]

    for feature in log_features:
        df = df.withColumn(f"{feature}_log", log1p(col(feature)))

    # Shift Monthly_Balance before log
    min_balance = df.agg({"Monthly_Balance": "min"}).first()[0]
    if min_balance is not None:
        df = df.withColumn("Monthly_Balance_log", log1p(col("Monthly_Balance") - lit(min_balance) + lit(1)))
    return df

def add_financial_ratios(df):
    # Avoid division by zero with when/otherwise
    df = df.withColumn(
        "Debt_to_Income_Ratio",
        when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income"))
        .otherwise(lit(0.0))
    )

    df = df.withColumn(
        "EMI_to_Income_Ratio",
        when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
        .otherwise(lit(0.0))
    )

    df = df.withColumn(
        "Invest_to_Income_Ratio",
        when(col("Monthly_Inhand_Salary") > 0, col("Amount_invested_monthly") / col("Monthly_Inhand_Salary"))
        .otherwise(lit(0.0))
    )

    df = df.withColumn(
        "Credit_Inquiry_to_Loan_Ratio",
        when(col("Num_of_Loan") > 0, col("Num_Credit_Inquiries") / col("Num_of_Loan"))
        .otherwise(lit(0.0))
    )

    return df

def process_gold_features_table(snapshot_date_str, silver_feature_directory, gold_feature_directory, spark):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Load silver features
    file_name = f"silver_feature_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    df = spark.read.parquet(os.path.join(silver_feature_directory, file_name))

    print("Loaded from:", file_name, "Row count:", df.count())

    # Type casting
    float_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Changed_Credit_Limit", "Credit_Utilization_Ratio", "Credit_History_Age"
    ]
    
    int_cols = [
        "Age", "Interest_Rate", "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Delay_from_due_date", "Num_Credit_Inquiries", "Num_Loan_Types"
    ]
    
    for col_name in float_cols:
        df = df.withColumn(col_name, col(col_name).cast(FloatType()))
    
    for col_name in int_cols:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

    # Outlier filtering
    df = df.filter(
        (col("Age").between(18, 100)) &
        (col("Interest_Rate").between(0, 100)) &
        (col("Num_Bank_Accounts").between(0, 50)) &
        (col("Num_Credit_Card").between(0, 50)) &
        (col("Num_of_Loan") <= 20) &
        (col("Num_of_Delayed_Payment") <= 100) &
        (col("Num_Credit_Inquiries") <= 1000) &
        (col("Total_EMI_per_month") < 100000) &
        (col("Amount_invested_monthly") < 100000) &
        (col("Monthly_Balance").between(-100000, 100000)) &
        (col("Annual_Income") < 1e7) &
        (col("Outstanding_Debt") < 1e7) &
        (col("Monthly_Inhand_Salary") < 1e6)
    )

    # Apply log transforms and ratios
    df = apply_log_transformations_spark(df)
    df = add_financial_ratios(df)

    # Drop multicolinear features
    multicollinear_raw_features = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Monthly_Balance',
        'Amount_invested_monthly', 'Outstanding_Debt', 'Total_EMI_per_month', 'Monthly_Inhand_Salary_log'
    ]
    
    df = df.drop(*multicollinear_raw_features)
    
    # Categorical cleaning
    df = df.withColumn("Occupation", when(col("Occupation") == "na", "unknown").otherwise(col("Occupation"))) \
           .withColumn("Credit_Mix", when(col("Credit_Mix") == "na", "unknown").otherwise(col("Credit_Mix"))) \
           .withColumn("Payment_of_Min_Amount", when(col("Payment_of_Min_Amount").isin("na", "nm"), "unknown").otherwise(col("Payment_of_Min_Amount"))) \
           .withColumn("Payment_Behaviour", when(col("Payment_Behaviour").isNull(), "unknown").otherwise(col("Payment_Behaviour"))) \
           .withColumn("Loan_Types", when(col("Loan_Types").isNull(), array().cast("array<string>")).otherwise(col("Loan_Types")))

    # Feature engineering
    df = df.withColumn("is_employed", when(col("Occupation") == "employed", 1).otherwise(0))

    # Save gold features
    output_file = f"gold_feature_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath = os.path.join(gold_feature_directory, output_file)
    df.write.mode("overwrite").parquet(filepath)
    print("Saved to:", filepath)

    return df