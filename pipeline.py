# ABC Retail Solutions ETL Pipeline

# Converted from pipeline.ipynb


# # ABC Retail Solutions ETL Pipeline
# 
# This notebook builds the complete retail data engineering workflow for the NeoStats assignment. It reads the raw Excel workbook, profiles the source data, cleans known data quality issues, merges and enriches the transaction records, calculates business KPIs, and exports the final reporting workbook.
# 
# The notebook is designed to run from top to bottom after a kernel restart.


# ## 0. Setup
# 
# This section imports libraries, finds the source workbook, and defines helper functions used throughout the notebook. The notebook expects the workbook to be stored at `data/retail_data.xlsx` inside the project folder, which keeps the project portable across computers.


from pathlib import Path
import hashlib

import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 120)


def find_project_root():
    """Find the folder that contains data/retail_data.xlsx, regardless of where Jupyter was opened."""
    current = Path.cwd().resolve()
    candidates = [current, *current.parents]

    for base in candidates:
        if (base / "data" / "retail_data.xlsx").exists():
            return base

    # Fallback for cases where Jupyter is launched from a parent folder.
    for match in current.glob("**/data/retail_data.xlsx"):
        return match.parent.parent

    raise FileNotFoundError(
        "Could not find data/retail_data.xlsx. Keep the notebook inside the project folder "
        "and make sure the raw workbook is stored in the data folder."
    )


PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "retail_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "clean_retail_data.xlsx"
EXPECTED_SHEETS = ["retail_data1", "retail_data2", "product_details"]

print("Project root detected successfully.")
print("Using source workbook: data/retail_data.xlsx")
print("Output workbook will be written to: Output/clean_retail_data.xlsx")


def print_step(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def show_shape(name, df):
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")


def summarize_nulls(df):
    return pd.DataFrame({
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
    })


def summarize_uniques(df):
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_count": df.nunique(dropna=False),
    })


# ## 1. Extract: Data Ingestion
# 
# The extract step reads all three Excel sheets into separate pandas DataFrames. No business logic is applied here. The goal is only to load the raw data and confirm that the expected sheets, row counts, columns, dtypes, and sample records are present.


workbook = pd.ExcelFile(DATA_PATH)
print(f"Workbook sheets found: {workbook.sheet_names}")

missing_sheets = sorted(set(EXPECTED_SHEETS) - set(workbook.sheet_names))
if missing_sheets:
    raise ValueError(f"Missing expected sheets: {missing_sheets}")

retail_data1 = pd.read_excel(DATA_PATH, sheet_name="retail_data1")
retail_data2 = pd.read_excel(DATA_PATH, sheet_name="retail_data2")
product_details = pd.read_excel(DATA_PATH, sheet_name="product_details")

source_tables = {
    "retail_data1": retail_data1,
    "retail_data2": retail_data2,
    "product_details": product_details,
}

for name, df in source_tables.items():
    print_step(f"Loaded {name}")
    show_shape(name, df)
    print("\nData types:")
    print(df.dtypes)
    print("\nSample rows:")
    display(df.drop(columns=["customer_name", "email", "phone"], errors="ignore").head())


# ## 2. Exploratory Data Analysis
# 
# This section checks the raw data before cleaning. It focuses on nulls, duplicate transaction IDs, unique counts, value distributions, mixed date formats, invalid quantities, inconsistent categories, inconsistent product names, and missing prices.


retail_all_raw = pd.concat(
    [
        retail_data1.assign(source="retail_data1"),
        retail_data2.assign(source="retail_data2"),
    ],
    ignore_index=True,
)

print_step("Combined raw retail data")
show_shape("retail_all_raw", retail_all_raw)

for name, df in source_tables.items():
    print_step(f"EDA summary: {name}")
    print("Null profile:")
    display(summarize_nulls(df))
    print("\nUnique-value profile:")
    display(summarize_uniques(df))
    print(f"Full duplicate rows: {df.duplicated().sum():,}")
    if "transaction_id" in df.columns:
        print(f"Duplicate transaction_id rows: {df.duplicated('transaction_id').sum():,}")


print_step("Key distributions in combined raw retail data")
for col in ["category", "city", "payment_method", "payment_status"]:
    print(f"{col}")
    display(retail_all_raw[col].value_counts(dropna=False))

print_step("Specific raw data quality checks")
date_type_counts = retail_all_raw["transaction_date"].map(lambda value: type(value).__name__).value_counts()
parsed_dates = pd.to_datetime(retail_all_raw["transaction_date"], errors="coerce")

raw_quality_findings = {
    "retail_data1_null_prices": int(retail_data1["price"].isna().sum()),
    "retail_data2_null_prices": int(retail_data2["price"].isna().sum()),
    "combined_null_prices": int(retail_all_raw["price"].isna().sum()),
    "retail_data1_duplicate_transaction_rows": int(retail_data1.duplicated("transaction_id").sum()),
    "retail_data2_duplicate_transaction_rows": int(retail_data2.duplicated("transaction_id").sum()),
    "combined_duplicate_transaction_rows": int(retail_all_raw.duplicated("transaction_id").sum()),
    "retail_data1_negative_quantity_rows": int((retail_data1["quantity"] < 0).sum()),
    "retail_data1_zero_quantity_rows": int((retail_data1["quantity"] == 0).sum()),
    "retail_data2_negative_quantity_rows": int((retail_data2["quantity"] < 0).sum()),
    "retail_data2_zero_quantity_rows": int((retail_data2["quantity"] == 0).sum()),
    "combined_quantity_lte_zero_rows": int((retail_all_raw["quantity"] <= 0).sum()),
    "category_raw_unique_values": int(retail_all_raw["category"].nunique(dropna=False)),
    "category_normalized_unique_values": int(retail_all_raw["category"].dropna().astype(str).str.strip().str.lower().nunique()),
    "product_name_raw_unique_values": int(retail_all_raw["product_name"].nunique(dropna=False)),
    "product_name_normalized_unique_values": int(retail_all_raw["product_name"].dropna().astype(str).str.strip().str.lower().nunique()),
    "unparseable_transaction_dates": int(parsed_dates.isna().sum()),
}

for key, value in raw_quality_findings.items():
    print(f"{key}: {value:,}")

print("\nTransaction date value types:")
display(date_type_counts)


# ### EDA Findings
# 
# The raw review found these issues: `retail_data1` has 404 null prices and `retail_data2` has 405 null prices. `retail_data1` has 243 duplicate `transaction_id` rows and `retail_data2` has 251. The combined raw data has 494 duplicate `transaction_id` rows. Quantity validation found 34 negative and 16 zero quantity rows in `retail_data1`, and 31 negative and 12 zero quantity rows in `retail_data2`. The `category` field has 12 raw variants that need to be mapped to four standard categories. The `product_name` field has 30 variants for 10 products. The date column contains mixed raw types, including Excel date values and string dates, and must be standardized before monthly analysis.


# ## 3. Transform: Cleaning
# 
# This section applies the cleaning rules in a fixed order. Each function returns a new DataFrame and prints what changed, so the row counts and issue resolution are easy to follow.


def remove_failed_duplicates(df):
    rows_before = len(df)
    duplicate_before = int(df.duplicated("transaction_id").sum())
    failed_before = int(df["payment_status"].astype(str).str.lower().eq("failed").sum())

    cleaned = df.copy()
    cleaned["_success_priority"] = cleaned["payment_status"].astype(str).str.lower().eq("successful").astype(int)
    cleaned = (
        cleaned.sort_values(["transaction_id", "_success_priority"], ascending=[True, False])
        .drop_duplicates("transaction_id", keep="first")
        .drop(columns="_success_priority")
        .reset_index(drop=True)
    )

    print_step("Remove failed duplicate transactions")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Rows removed: {rows_before - len(cleaned):,}")
    print(f"Duplicate transaction_id rows before: {duplicate_before:,}")
    print(f"Duplicate transaction_id rows after:  {cleaned.duplicated('transaction_id').sum():,}")
    print(f"Failed rows before: {failed_before:,}")
    print(f"Failed rows after:  {cleaned['payment_status'].astype(str).str.lower().eq('failed').sum():,}")
    return cleaned


def fix_date_formats(df):
    rows_before = len(df)
    cleaned = df.copy()

    def parse_date(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(value, errors="coerce")

    cleaned["transaction_date"] = cleaned["transaction_date"].apply(parse_date)
    cleaned["transaction_date"] = pd.to_datetime(cleaned["transaction_date"], errors="coerce").dt.normalize()

    print_step("Fix transaction date formats")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Unparseable dates after: {cleaned['transaction_date'].isna().sum():,}")
    print("Sample dates:")
    display(cleaned["transaction_date"].dt.strftime("%Y-%m-%d").head())
    return cleaned


def fill_missing_prices(df, product_details):
    rows_before = len(df)
    missing_before = int(df["price"].isna().sum())
    cleaned = df.copy()
    price_lookup = product_details.set_index("product_id")["price"]
    cleaned["price"] = cleaned["price"].fillna(cleaned["product_id"].map(price_lookup))

    print_step("Fill missing prices")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Missing prices before: {missing_before:,}")
    print(f"Missing prices after:  {cleaned['price'].isna().sum():,}")
    return cleaned


def remove_invalid_quantities(df):
    rows_before = len(df)
    invalid_before = int((df["quantity"] <= 0).sum())
    negative_before = int((df["quantity"] < 0).sum())
    zero_before = int((df["quantity"] == 0).sum())
    cleaned = df[df["quantity"] > 0].copy().reset_index(drop=True)

    print_step("Remove invalid quantities")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Rows removed: {rows_before - len(cleaned):,}")
    print(f"Invalid rows before: {invalid_before:,} ({negative_before:,} negative, {zero_before:,} zero)")
    print(f"Invalid rows after:  {(cleaned['quantity'] <= 0).sum():,}")
    return cleaned


def standardize_categories(df):
    rows_before = len(df)
    unique_before = int(df["category"].nunique(dropna=False))
    cleaned = df.copy()
    category_map = {
        "elec": "Electronics",
        "electronics": "Electronics",
        "cloth": "Clothing",
        "clothing": "Clothing",
        "furn": "Furniture",
        "furniture": "Furniture",
        "home": "Home Appliances",
        "home appliances": "Home Appliances",
    }
    cleaned["category"] = cleaned["category"].astype(str).str.strip().str.lower().map(category_map)

    print_step("Standardize categories")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Unique categories before: {unique_before:,}")
    print(f"Unique categories after:  {cleaned['category'].nunique(dropna=False):,}")
    print(f"Unmapped categories after: {cleaned['category'].isna().sum():,}")
    display(cleaned["category"].value_counts())
    return cleaned


def standardize_product_names(df, product_details):
    rows_before = len(df)
    unique_before = int(df["product_name"].nunique(dropna=False))
    cleaned = df.copy()
    name_lookup = product_details.set_index("product_id")["product_name"]
    cleaned["product_name"] = cleaned["product_id"].map(name_lookup)

    print_step("Standardize product names")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print(f"Unique product names before: {unique_before:,}")
    print(f"Unique product names after:  {cleaned['product_name'].nunique(dropna=False):,}")
    print(f"Missing product names after: {cleaned['product_name'].isna().sum():,}")
    return cleaned


def mask_pii(df):
    rows_before = len(df)
    cleaned = df.copy()

    def sha256_hash(value):
        if pd.isna(value):
            return pd.NA
        return hashlib.sha256(str(value).strip().lower().encode("utf-8")).hexdigest()

    cleaned["email"] = cleaned["email"].apply(sha256_hash)
    cleaned["phone"] = cleaned["phone"].apply(sha256_hash)

    print_step("Mask email and phone")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {len(cleaned):,}")
    print("Email hash lengths:", set(cleaned["email"].dropna().astype(str).str.len()))
    print("Phone hash lengths:", set(cleaned["phone"].dropna().astype(str).str.len()))
    return cleaned


cleaned_retail = retail_all_raw.copy()
cleaned_retail = remove_failed_duplicates(cleaned_retail)
cleaned_retail = fix_date_formats(cleaned_retail)
cleaned_retail = fill_missing_prices(cleaned_retail, product_details)
cleaned_retail = remove_invalid_quantities(cleaned_retail)
cleaned_retail = standardize_categories(cleaned_retail)
cleaned_retail = standardize_product_names(cleaned_retail, product_details)
cleaned_retail = mask_pii(cleaned_retail)

print_step("Cleaning validation")
validation = {
    "final_rows": len(cleaned_retail),
    "duplicate_transaction_id_rows": int(cleaned_retail.duplicated("transaction_id").sum()),
    "failed_payment_rows": int(cleaned_retail["payment_status"].astype(str).str.lower().eq("failed").sum()),
    "missing_price_rows": int(cleaned_retail["price"].isna().sum()),
    "quantity_lte_zero_rows": int((cleaned_retail["quantity"] <= 0).sum()),
    "missing_category_rows": int(cleaned_retail["category"].isna().sum()),
    "missing_product_name_rows": int(cleaned_retail["product_name"].isna().sum()),
}
display(pd.DataFrame.from_dict(validation, orient="index", columns=["value"]))

assert validation["duplicate_transaction_id_rows"] == 0
assert validation["failed_payment_rows"] == 0
assert validation["missing_price_rows"] == 0
assert validation["quantity_lte_zero_rows"] == 0
assert validation["missing_category_rows"] == 0
assert validation["missing_product_name_rows"] == 0


# ## 4. Transform: Merge and Enrich
# 
# The cleaned records are split by their original source and merged back together with a `source` column. Then the data is enriched with product reference fields from `product_details`.


def merge_datasets(df1, df2):
    left = df1.copy()
    right = df2.copy()
    left["source"] = "retail_data1"
    right["source"] = "retail_data2"
    merged = pd.concat([left, right], ignore_index=True)
    print_step("Merge cleaned retail extracts")
    print(f"retail_data1 rows: {len(left):,}")
    print(f"retail_data2 rows: {len(right):,}")
    print(f"Merged rows: {len(merged):,}")
    return merged


def validate_merge(df_merged, expected_rows):
    report = {
        "expected_rows": expected_rows,
        "actual_rows": len(df_merged),
        "rows_lost": expected_rows - len(df_merged),
        "source_null_rows": int(df_merged["source"].isna().sum()),
        "source_values": df_merged["source"].value_counts().to_dict(),
    }
    print_step("Merge validation")
    display(pd.DataFrame.from_dict(report, orient="index", columns=["value"]))
    assert report["rows_lost"] == 0
    assert report["source_null_rows"] == 0
    return report


def enrich_with_product_details(df_merged, product_details):
    product_ref = product_details[["product_id", "category", "price"]].rename(
        columns={"category": "std_category", "price": "std_price"}
    )
    enriched = df_merged.merge(product_ref, on="product_id", how="left")
    print_step("Enrich with product_details")
    print(f"Rows before: {len(df_merged):,}")
    print(f"Rows after:  {len(enriched):,}")
    return enriched


def validate_enrichment(df_enriched):
    unmatched = df_enriched[df_enriched["std_category"].isna() | df_enriched["std_price"].isna()]
    report = {
        "rows": len(df_enriched),
        "unmatched_rows": len(unmatched),
        "unmatched_product_ids": sorted(unmatched["product_id"].dropna().unique().tolist()),
        "std_category_null_rows": int(df_enriched["std_category"].isna().sum()),
        "std_price_null_rows": int(df_enriched["std_price"].isna().sum()),
    }
    print_step("Enrichment validation")
    display(pd.DataFrame.from_dict(report, orient="index", columns=["value"]))
    assert report["unmatched_rows"] == 0
    return report

cleaned_retail_data1 = cleaned_retail[cleaned_retail["source"] == "retail_data1"].drop(columns="source")
cleaned_retail_data2 = cleaned_retail[cleaned_retail["source"] == "retail_data2"].drop(columns="source")

merged_retail = merge_datasets(cleaned_retail_data1, cleaned_retail_data2)
merge_report = validate_merge(merged_retail, expected_rows=len(cleaned_retail))
enriched_retail = enrich_with_product_details(merged_retail, product_details)
enrichment_report = validate_enrichment(enriched_retail)

print_step("Enriched dataset preview")
show_shape("enriched_retail", enriched_retail)
print(list(enriched_retail.columns))
display(enriched_retail.drop(columns=["customer_name", "email", "phone"], errors="ignore").head())


# ## 5. Load: KPI Aggregation and Export
# 
# The load step calculates revenue and KPI tables from the clean enriched data. The clean data and all KPI outputs are exported to `clean_retail_data.xlsx` for Power BI.


def calculate_revenue(df):
    result = df.copy()
    result["revenue"] = result["quantity"] * result["price"] * (1 - result["discount"])
    result["revenue"] = result["revenue"].round(2)
    result["transaction_month"] = result["transaction_date"].dt.to_period("M").astype(str)
    print_step("Calculate revenue")
    print("Formula: revenue = quantity x price x (1 - discount)")
    print(f"Total revenue: ${result['revenue'].sum():,.2f}")
    return result

clean_data = calculate_revenue(enriched_retail)

total_revenue = clean_data["revenue"].sum()
transaction_count = clean_data["transaction_id"].nunique()
avg_order_value = total_revenue / transaction_count

kpi_total_revenue = pd.DataFrame({
    "metric": ["Total Revenue"],
    "value": [round(total_revenue, 2)],
    "formatted_value": [f"${total_revenue:,.2f}"],
})

kpi_revenue_by_category = clean_data.groupby("category", as_index=False).agg(
    revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("transaction_id", "nunique")
).sort_values("revenue", ascending=False).reset_index(drop=True)

kpi_revenue_by_city = clean_data.groupby("city", as_index=False).agg(
    revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("transaction_id", "nunique")
).sort_values("revenue", ascending=False).reset_index(drop=True)

kpi_monthly_trend = clean_data.groupby("transaction_month", as_index=False).agg(
    revenue=("revenue", "sum"), transactions=("transaction_id", "nunique")
).sort_values("transaction_month").reset_index(drop=True)

kpi_top5_products = clean_data.groupby("product_name", as_index=False).agg(
    revenue=("revenue", "sum"), quantity=("quantity", "sum"), avg_discount=("discount", "mean")
).sort_values("revenue", ascending=False).head(5).reset_index(drop=True)

kpi_by_payment_method = clean_data.groupby("payment_method", as_index=False).agg(
    revenue=("revenue", "sum"), transactions=("transaction_id", "nunique")
).sort_values("revenue", ascending=False).reset_index(drop=True)

kpi_online_vs_offline = clean_data.groupby("purchase_location", as_index=False).agg(
    revenue=("revenue", "sum"), transactions=("transaction_id", "nunique")
).sort_values("revenue", ascending=False).reset_index(drop=True)

kpi_avg_order_value = pd.DataFrame({
    "metric": ["Average Order Value"],
    "total_revenue": [round(total_revenue, 2)],
    "transaction_count": [transaction_count],
    "value": [round(avg_order_value, 2)],
    "formatted_value": [f"${avg_order_value:,.2f}"],
})

for df in [kpi_revenue_by_category, kpi_revenue_by_city, kpi_monthly_trend, kpi_top5_products, kpi_by_payment_method, kpi_online_vs_offline]:
    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].round(2)
if "avg_discount" in kpi_top5_products.columns:
    kpi_top5_products["avg_discount"] = kpi_top5_products["avg_discount"].round(4)


kpis = {
    "kpi_total_revenue": kpi_total_revenue,
    "kpi_revenue_by_category": kpi_revenue_by_category,
    "kpi_revenue_by_city": kpi_revenue_by_city,
    "kpi_monthly_trend": kpi_monthly_trend,
    "kpi_top5_products": kpi_top5_products,
    "kpi_by_payment_method": kpi_by_payment_method,
    "kpi_online_vs_offline": kpi_online_vs_offline,
    "kpi_avg_order_value": kpi_avg_order_value,
}

for name, df in kpis.items():
    print_step(name)
    display(df)


def export_clean_data(df, kpis, output_path=OUTPUT_PATH):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="clean_data", index=False)
        for sheet_name, kpi_df in kpis.items():
            kpi_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print_step("Export complete")
    print("Exported: Output/clean_retail_data.xlsx")
    print("Sheets:", ["clean_data"] + list(kpis.keys()))
    return output_path

export_clean_data(clean_data, kpis)


# ## 6. Business Insights
# 
# The cleaned dataset contains 7,914 valid transactions and generates total revenue of **$1,163,993,285.00**. The average order value is **$147,080.27**, calculated from total revenue divided by the number of valid transactions.
# 
# **Electronics** is the strongest category with **$674,462,500.00** in revenue from **2,395** transactions. This is much higher than Furniture at **$243,413,000.00** and Home Appliances at **$229,025,250.00**, so Electronics is the main revenue category.
# 
# **Chennai** is the top city with **$246,028,300.00** in revenue, followed by Delhi at **$242,022,530.00** and Hyderabad at **$232,881,480.00**. The city revenue spread is fairly close, so the dashboard should compare city performance by product category and channel.
# 
# **Laptop** is the highest-revenue product at **$466,250,000.00**. The next highest products are Sofa at **$150,708,000.00**, Phone at **$127,568,000.00**, Refrigerator at **$117,975,000.00**, and Dining Table at **$92,705,000.00**.
# 
# Online and offline revenue are almost balanced. Online purchases generated **$584,104,340.00**, while offline purchases generated **$579,888,945.00**. Both channels should be included prominently in the Power BI dashboard.
