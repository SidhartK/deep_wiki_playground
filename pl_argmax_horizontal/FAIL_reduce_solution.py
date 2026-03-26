import polars as pl  
  
df = pl.DataFrame({  
    "a": [1, 5, 3],  
    "b": [4, 2, 6],   
    "c": [2, 8, 1]  
})  
  
col_names = df.columns  
  
# Use reduce with a custom function that tracks (value, index)  
argmax_result = df.select(  
    pl.reduce(  
        lambda acc, x: pl.when(x > acc[0]).then((x, acc[1] + 1)).otherwise(acc),  
        [(pl.col(col), pl.lit(i)) for i, col in enumerate(col_names)]  
    ).struct[1].map_dict(  
        {i: name for i, name in enumerate(col_names)},  
        default=pl.first()  
    ).alias("argmax_col")  
)