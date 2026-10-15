# Week 2: Data Cleaning and Preprocessing for Logistics Analysis

## Project Overview

This project focuses on data collection, cleaning, and preprocessing for logistics analysis as part of the Logistics Data Analyst Internship at YuvaIntern.

The objective is to prepare logistics data for reliable analysis by identifying missing values, removing duplicate records, detecting outliers, and normalizing numerical features. These preprocessing steps help improve data quality and prepare the dataset for future analysis and predictive modeling.

## Dataset

**Reference Dataset:** DataCo Smart Supply Chain Dataset

The dataset is intended to support the analysis of logistics and supply chain operations. Before processing, the actual CSV file and its column names must be verified.

**Note:** The actual dataset must be downloaded and processed before reporting results as findings from DataCo. Any synthetic examples used for demonstration are not actual DataCo records.

## Objectives

- Inspect the dataset structure and data types.
- Identify missing values in relevant numerical columns.
- Remove duplicate records.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Handle missing numerical values using median imputation.
- Normalize numerical features using Min-Max scaling.
- Export cleaned data and an outlier audit for further analysis.

## Project Files

- `Task_2_Data_Cleaning_Preprocessing_Report.docx` — Documents the objectives, methodology, calculations, preprocessing techniques, and limitations.
- `preprocessing_pipeline.py` — Python script demonstrating the data inspection, cleaning, outlier detection, normalization, and export workflow.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Methodology

1. Load the dataset and inspect its columns, data types, and missing values.
2. Remove exact duplicate rows.
3. Convert relevant numerical columns to numeric data types.
4. Fill missing numerical values using the median.
5. Identify potential outliers using the IQR method and record them for review.
6. Apply IQR-based capping where appropriate.
7. Normalize selected numerical features using Min-Max scaling.
8.
