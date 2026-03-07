import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    result = np.zeros((seq_len, d_model))

    positions = np.arange(0, seq_len)[:, None]
    div = np.power(base, np.arange(0, d_model, 2)/ d_model)

    result[:, ::2] = np.sin(positions / div)
    result[:, 1::2] = np.cos(positions / div)[:, :d_model//2]
    return result
    