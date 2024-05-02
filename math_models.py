import math
import numpy as np
from numba import njit

def trans_rate(B, tx_power_dbm, noise_power_dbm, H, BF) -> float:
    x = np.abs((H*BF).sum())
    x = dec_to_db(x**2)
    x = tx_power_dbm + x - noise_power_dbm - dec_to_db(B) - 60 - 3
    x = db_to_dec(x)
    # return B * math.log2(1 + x)
    return math.log2(1 + x)


@njit(fastmath=True, parallel=False)
def dec_to_db(dec: float) -> float:
    return 10 * math.log10(dec)

@njit(fastmath=True, parallel=False)
def db_to_dec(db: float) -> float:
    return 10 ** (db / 10)


if __name__ == '__main__':

    import timeit
    dec_time = timeit.timeit(stmt='dec_to_db(11.2)',
    setup='from __main__ import dec_to_db') 
    print(dec_time)
