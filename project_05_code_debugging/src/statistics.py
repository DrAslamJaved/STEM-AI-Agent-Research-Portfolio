def calculate_mean(data):
    """
    Calculate the arithmetic mean of a non-empty dataset.

    Parameters
    ----------
    data : iterable of numbers
        Numerical observations.

    Returns
    -------
    float
        Arithmetic mean of the data.

    Raises
    ------
    ValueError
        If the dataset is empty.
    TypeError
        If the input contains non-numeric values.
    """

    if not data:
        raise ValueError("Dataset cannot be empty.")

    try:
        return sum(data) / len(data)
    except TypeError as exc:
        raise TypeError("Dataset must contain numeric values.") from exc