def get_target_column(df):
    """
    Automatically detect the target ASD column in the dataset
    """

    possible_targets = [
        "Class/ASD",
        "Class_ASD",
        "ASD",
        "ASD_Class",
        "Outcome",
        "target",
        "result"
    ]

    # Exact match first
    for col in possible_targets:
        if col in df.columns:
            return col

    # Fallback: partial match
    for col in df.columns:
        col_lower = col.lower()
        if "asd" in col_lower or "class" in col_lower or "target" in col_lower:
            return col

    raise ValueError(
        "Target ASD column not found in dataset. "
        "Please check column names."
    )
