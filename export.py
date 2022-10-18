#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

from ztfnuclear.database import MetadataDB
import os, logging, re, json

meta = MetadataDB()

params = ["comments", "rating"]

for p in params:
    content = meta.read_parameters(["_id", p])
    outfile = f"{p}.json"

    with open(outfile, "w") as fp:
        json.dump(content, fp)
