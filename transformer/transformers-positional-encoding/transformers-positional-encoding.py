import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    base = 10000

    result = np.zeros((seq_length, d_model))

    pos = np.arange(0, seq_length)[:, None]

    div = np.power(1000, np.arange(0, d_model, 2) / d_model)

    result[:, ::2] = np.sin(pos / div)
    result[:, 1::2] = np.cos(pos / div)[:d_model // 2]
    
    return result