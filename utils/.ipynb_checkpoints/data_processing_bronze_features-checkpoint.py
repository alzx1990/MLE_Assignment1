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

from pyspark.sql.functions import col, to_date, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_features_table(snapshot_date_str, bronze_feature_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    clickstream_path = "data/feature_clickstream.csv"
    attributes_path = "data/features_attributes.csv"
    financials_path = "data/features_financials.csv"

    # Clickstream - filter by snapshot
    # df_clickstream = spark.read.csv(clickstream_path, header=True, inferSchema=True).filter(col("snapshot_date") == snapshot_date_str)
    df_clickstream = spark.read.csv(clickstream_path, header=True, inferSchema=True)
    df_clickstream = df_clickstream.withColumn("snapshot_date", to_date(col("snapshot_date")))
    df_clickstream = df_clickstream.filter(col("snapshot_date") == to_date(lit(snapshot_date_str), "yyyy-MM-dd"))
    print(snapshot_date_str + 'row count:', df_clickstream.count())

    # Attributes - filter by snapshot
    # df_attributes = spark.read.csv(attributes_path, header=True, inferSchema=True).filter(col("snapshot_date") == snapshot_date_str)
    df_attributes = spark.read.csv(attributes_path, header=True, inferSchema=True)
    df_attributes = df_attributes.withColumn("snapshot_date", to_date(col("snapshot_date")))
    df_attributes = df_attributes.filter(col("snapshot_date") == to_date(lit(snapshot_date_str), "yyyy-MM-dd"))
    print(snapshot_date_str + 'row count:', df_attributes.count())

    # Financials - assumed static (no snapshot column)
    # df_financials = spark.read.csv(financials_path, header=True, inferSchema=True).filter(col("snapshot_date") == snapshot_date_str)
    df_financials = spark.read.csv(financials_path, header=True, inferSchema=True)
    df_financials = df_financials.withColumn("snapshot_date", to_date(col("snapshot_date")))
    df_financials = df_financials.filter(col("snapshot_date") == to_date(lit(snapshot_date_str), "yyyy-MM-dd"))
    print(snapshot_date_str + 'row count:', df_financials.count())

    # save bronze table to datamart - IRL connect to database to write
    for name, df in [("clickstream", df_clickstream), ("attributes", df_attributes), ("financials", df_financials)]:
        filename = f"bronze_feature_{name}_" + snapshot_date_str.replace('-', '_') + ".csv"
        filepath = os.path.join(bronze_feature_directory, filename)
        df.toPandas().to_csv(filepath, index=False)
        print("saved to:", filepath)

    return df_clickstream, df_attributes, df_financials
