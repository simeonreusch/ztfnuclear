#!/usr/bin/env python3
# The coded used here was developed by Valéry Brinnel
# License: BSD-3-Clause

import os, logging

alphabet = "abcdefghijklmnopqrstuvwxyz"
rg = (6, 5, 4, 3, 2, 1, 0)
dec_ztf_years = {i: str(i + 17) for i in range(16)}


def stockid_to_ztfid(stockid: int) -> str:
    """Converts AMPEL internal stock ID to ZTF ID"""
    year = dec_ztf_years[stockid & 15]

    # Shift base10 encoded value 4 bits to the right
    stockid = stockid >> 4

    # Convert back to base26
    l = ["a", "a", "a", "a", "a", "a", "a"]
    for i in rg:
        l[i] = alphabet[stockid % 26]
        stockid //= 26
        if not stockid:
            break

    return f"ZTF{year}{''.join(l)}"
