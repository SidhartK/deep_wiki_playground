from __future__ import annotations  
  
from typing import Union  
import polars as pl  
  
  
def argmax_horizontal(  
    *exprs: Union[str, pl.Expr],  
    return_names: bool = False,  
) -> pl.Expr:  
    """  
    Row-wise argmax across columns — analogous to pl.max_horizontal but  
    returning the position of the maximum rather than the maximum value.  
  
    Parameters  
    ----------  
    *exprs  
        Column names (str) or Expr objects to consider. All must share a  
        common numeric supertype (same constraint as pl.max_horizontal).  
    return_names  
        If False (default), return the 0-based column index as  
        pl.get_index_type() (UInt32 or UInt64 depending on build).  
        If True, return the winning column's name as pl.String.  
  
    Returns  
    -------  
    pl.Expr  
        Per-row index or name of the column holding the maximum value.  
        Yields null when ALL inputs for that row are null.  
  
    Notes  
    -----  
    Null handling mirrors pl.max_horizontal exactly:  
      - Null values in individual columns are IGNORED when finding the max.  
      - A row is null in the output only when every input column is null.  
    Tie-breaking: the FIRST (lowest-index) column wins on equal values,  
    consistent with numpy.argmax / list.arg_max behaviour.  
  
    Examples  
    --------  
    >>> df = pl.DataFrame({  
    ...     "a": [1,   8,    3,    None],  
    ...     "b": [4,   5,    None, None],  
    ...     "c": [2,   6,    7,    None],  
    ... })  
    >>> df.with_columns(  
    ...     argmax_idx  = argmax_horizontal("a", "b", "c"),  
    ...     argmax_name = argmax_horizontal("a", "b", "c", return_names=True),  
    ... )  
    shape: (4, 5)  
    ┌──────┬──────┬──────┬─────────────┬─────────────┐  
    │ a    ┆ b    ┆ c    ┆ argmax_idx  ┆ argmax_name │  
    │ ---  ┆ ---  ┆ ---  ┆ ---         ┆ ---         │  
    │ i64  ┆ i64  ┆ i64  ┆ u32         ┆ str         │  
    ╞══════╪══════╪══════╪═════════════╪═════════════╡  
    │ 1    ┆ 4    ┆ 2    ┆ 1           ┆ b           │  
    │ 8    ┆ 5    ┆ 6    ┆ 0           ┆ a           │  
    │ 3    ┆ null ┆ 7    ┆ 2           ┆ c           │  
    │ null ┆ null ┆ null ┆ null        ┆ null        │  
    └──────┴──────┴──────┴─────────────┴─────────────┘  
    """  
    if not exprs:  
        msg = "argmax_horizontal requires at least one expression"  
        raise ValueError(msg)  
  
    # Resolve to Expr objects (strings → col references)  
    parsed: list[pl.Expr] = [  
        pl.col(e) if isinstance(e, str) else e for e in exprs  
    ]  
  
    # ------------------------------------------------------------------ #  
    # Core computation                                                      #  
    #                                                                       #  
    # pl.concat_list wraps each scalar value into a 1-element list and     #  
    # concatenates them row-wise.  A null scalar becomes a null ELEMENT     #  
    # inside the list (not a null list row), so list.arg_max() ignores it   #  
    # exactly the way max_horizontal ignores null column values.            #  
    # ------------------------------------------------------------------ #  
    idx_expr: pl.Expr = pl.concat_list(*parsed).list.arg_max()  
  
    if not return_names:  
        return idx_expr  
  
    # ------------------------------------------------------------------ #  
    # Optional: map the integer index back to a column name               #  
    # We use the chained when-then API to build a per-row lookup.         #  
    # ------------------------------------------------------------------ #  
    col_names: list[str] = [  
        e if isinstance(e, str) else e.meta.output_name()  
        for e in exprs  
    ]  
  
    # Build:  when(idx==0).then("a").when(idx==1).then("b")...otherwise(None)  
    chain = pl.when(idx_expr == 0).then(pl.lit(col_names[0]))  
    for i, name in enumerate(col_names[1:], start=1):  
        chain = chain.when(idx_expr == i).then(pl.lit(name))  
  
    return chain.otherwise(pl.lit(None, dtype=pl.String))

if __name__ == "__main__":
    df = pl.DataFrame({
        "a": [1, 5, 3],
        "b": [4, 2, 6],
        "c": [2, 8, 1]
    })
    print(df)
    print(df.select(argmax_horizontal("a", "b", "c")))
