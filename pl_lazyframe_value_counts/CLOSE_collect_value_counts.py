import polars as pl  
  
# Create a lazy frame  
lf = pl.LazyFrame({"column_name": ["a", "b", "a", "c", "b", "a"]})  
  
# Get value counts  
result = lf.select(pl.col("column_name").value_counts()).collect()

print(result)