import polars as pl  

df = pl.DataFrame({  
    "a": [1, 5, 3],  
    "b": [4, 2, 6],   
    "c": [2, 8, 1]  
})  
  
# Get column names  
col_names = df.columns  
  
# Compute argmax_horizontal  
argmax_result = df.select(  
    pl.max_horizontal(pl.all().alias("max_val")).alias("max_val"),  
    pl.struct([  
        pl.col(name).alias(name) for name in col_names  
    ]).map_elements(  
        lambda row: col_names[row.to_struct().field_names().index(  
            max(row.to_struct().values(), key=lambda x: x if x is not None else float('-inf'))  
        )],  
        return_dtype=pl.Utf8  
    ).alias("argmax_col")  
)