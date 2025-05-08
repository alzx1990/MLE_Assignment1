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

from pyspark.sql.functions import col, to_date, lit, regexp_replace, regexp_extract, regexp_replace, when, lower, trim, split, size, expr
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def enforce_column_types(df):
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": FloatType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": StringType(),  # keep for tracking, drop in gold
    }

    # Add fe_1 to fe_20 from clickstream
    for i in range(1, 21):
        column_type_map[f"fe_{i}"] = FloatType()

    for col_name, col_type in column_type_map.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(col_type))

    return df

def clean_and_cast_numeric(df, column_names):
    for col_name in column_names:
        clean_col = col_name + "_clean"
        
        # Remove non-numeric characters
        # df = df.withColumn(clean_col, regexp_replace(col(col_name), "[^0-9.]", ""))
        df = df.withColumn(clean_col, regexp_extract(col(col_name), r"-?\d+\.?\d*", 0))

        # Cast to float
        df = df.withColumn(clean_col, col(clean_col).cast("float"))

        #  Count number of cleaned rows
        cleaned_diff = df.filter(
            (col(col_name).isNotNull()) &
            (
                col(col_name).cast("float").isNull()  # original invalid
                | (col(col_name).cast("float") != col(clean_col))  # or value changed
            )
        ).count()

        print(f"Cleaned {cleaned_diff} row(s) in column: {col_name}")

        # Drop original and rename
        df = df.drop(col_name).withColumnRenamed(clean_col, col_name)

        print(f"Final null count for {col_name}: {df.filter(col(col_name).isNull()).count()}")
    return df

def clean_categorical_columns(df, categorical_cols):
    """
    Clean categorical columns by:
    - Lowercasing
    - Trimming whitespace
    - Replacing nulls, blanks, and symbol-only values with 'na'
    """
    for col_name in categorical_cols:
        df = df.withColumn(
            col_name,
            when(
                col(col_name).isNull() | (trim(col(col_name)) == "") |
                (regexp_extract(col(col_name), r"^[^a-zA-Z0-9]+$", 0) != ""),  # only symbols
                "na"
            ).otherwise(trim(lower(col(col_name))))
        )
    return df

def clean_credit_history_age(df):
    # Extract "XX Years and YY Months" → total months
    df = df.withColumn("years", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast("int"))
    df = df.withColumn("months", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast("int"))
    df = df.withColumn("Credit_History_Age", (col("years") * 12 + col("months")).cast("int"))
    return df.drop("years", "months")


def process_type_of_loan(df):
    # Normalize (lowercase, replace " and " with ",")
    df = df.withColumn("Loan_Types_raw", regexp_replace(lower(col("Type_of_Loan")), r"\s+and\s+", ","))

    # Split into array
    df = df.withColumn("Loan_Types_array", split(col("Loan_Types_raw"), ","))

    # Trim elements
    df = df.withColumn("Loan_Types_trimmed", expr("filter(transform(Loan_Types_array, x -> trim(x)), x -> x != '')"))

    # Filter out 'not specified'
    df = df.withColumn("Loan_Types_filtered", expr(
        "filter(Loan_Types_trimmed, x -> x != 'not specified')"
    ))

    # Remove duplicates
    df = df.withColumn("Loan_Types", expr("array_distinct(Loan_Types_filtered)"))

    # Add numeric feature
    df = df.withColumn("Num_Loan_Types", when(col("Loan_Types").isNotNull(), expr("size(Loan_Types)")).otherwise(0))

    # Drop intermediate columns
    df = df.drop("Loan_Types_raw", "Loan_Types_array", "Loan_Types_trimmed", "Loan_Types_filtered")

    return df

def process_payment_behaviour(df):
    valid_behaviours = [
        "low_spent_small_value_payments",
        "low_spent_medium_value_payments",
        "low_spent_high_value_payments",
        "high_spent_small_value_payments",
        "high_spent_medium_value_payments",
        "high_spent_high_value_payments"
    ]
    df = df.withColumn("Payment_Behaviour", trim(lower(col("Payment_Behaviour"))))
    df = df.withColumn("Payment_Behaviour", when(
        col("Payment_Behaviour").isin(valid_behaviours),
        col("Payment_Behaviour")
    ).otherwise("unknown"))
    return df

def process_silver_features_table(snapshot_date_str, bronze_feature_directory, silver_feature_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Load bronze CSVs
    def load_and_filter_bronze(name):
        filename = f"bronze_feature_{name}_" + snapshot_date_str.replace('-', '_') + ".csv"
        df = spark.read.csv(os.path.join(bronze_feature_directory, filename), header=True, inferSchema=True)
        df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
        return df.filter(col("snapshot_date") == to_date(lit(snapshot_date_str), "yyyy-MM-dd"))
        
    df_clickstream = load_and_filter_bronze("clickstream")
    df_attributes = load_and_filter_bronze("attributes")
    df_financials = load_and_filter_bronze("financials")

    # Log row counts
    print("Rows: clickstream =", df_clickstream.count(), 
          "| attributes =", df_attributes.count(), 
          "| financials =", df_financials.count())
    
    # Drop useless columns
    df_clickstream = df_clickstream.drop("snapshot_date")  # already filtered
    df_attributes = df_attributes.drop("snapshot_date", "Name", "SSN")
    df_financials = df_financials.drop("snapshot_date")

    # Clean Customer_ID across all sources
    df_clickstream = df_clickstream.withColumn("Customer_ID", trim(lower(col("Customer_ID"))))
    df_attributes  = df_attributes.withColumn("Customer_ID", trim(lower(col("Customer_ID"))))
    df_financials  = df_financials.withColumn("Customer_ID", trim(lower(col("Customer_ID"))))
    
    # Join all on Customer_ID
    df_joined = df_attributes \
        .join(df_financials, on="Customer_ID", how="left") \
        .join(df_clickstream, on="Customer_ID", how="left")

    print("✅ Joined row count:", df_joined.count())

    # Clean up numeric columns
    columns_to_clean = [
"Age", "Monthly_Inhand_Salary", "Annual_Income", "Interest_Rate", "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
"Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
"Amount_invested_monthly", "Monthly_Balance"
    ]
    
    df_joined = clean_and_cast_numeric(df_joined, columns_to_clean)
    df_joined = clean_credit_history_age(df_joined)

    # Clean specific complex categoricals
    df_joined = process_type_of_loan(df_joined)
    df_joined = process_payment_behaviour(df_joined)

    # Normalize categoricals
    categorical_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount"]
    df_joined = clean_categorical_columns(df_joined, categorical_cols)
    
    # Enforce schema and cast types
    df_joined = enforce_column_types(df_joined)
    # print("Final schema:")
    # df_joined.printSchema()

    # Save silver table - IRL connect to database to write
    output_file = f"silver_feature_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath = os.path.join(silver_feature_directory, output_file)
    df_joined.write.mode("overwrite").parquet(filepath)
    print("Saved to:", filepath, "Row count:", df_joined.count())
    
    return df_joined