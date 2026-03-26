import polars as pl  
  
df = pl.DataFrame({  
    "a": [1, 5, 3],  
    "b": [4, 2, 6],   
    "c": [2, 8, 1]  
})  
  
print(df)
# Get column names  
col_names = df.columns  
  
# Compute argmax_horizontal efficiently  
argmax_result = df.select(  
    *[  
        (pl.col(name) == pl.max_horizontal(pl.all())).alias(name)  
        for name in col_names  
    ]  
).select(  
    pl.coalesce([  
        pl.when(pl.col(name)).then(pl.lit(name))  
        for name in col_names  
    ]).alias("argmax_col")  
)

print(argmax_result)