import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    result = np.zeros((seq_len, d_model))
    
    pos = np.arange(seq_len)[:, None]

    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)


    result[:, 0::2] = np.sin(pos / div_term)
    
    if d_model % 2 == 0:
    

    
        result[:, 1::2] = np.cos(pos / div_term)
    else:
        result[:, 1::2] = np.cos(pos / div_term[:-1])
        
    return result