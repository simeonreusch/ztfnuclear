#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import logging

import pandas as pd

from ztfnuclear import io
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.sample import NuclearSample, Transient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.database").setLevel(logging.INFO)
logging.getLogger("ztfnuclear.sample").setLevel(logging.INFO)

# si = SampleInfo()
# meta = MetadataDB()
# s = NuclearSample()

# s.generate_ir_flare_sample()
# s.generate_overview_pickled(flaring_only=True)

infile = "/Users/simeon/ZTFDATA/nuclear_sample/plots/maghist/surviving_tde_cleaned.csv"
df = pd.read_csv(infile)


s = NuclearSample()
# t = Transient("ZTF20abgxlut")
# t.plot(xlim=[58750, 60000], wide=True)

for ztfid in df.ztfid.values:
    t = Transient(ztfid)
    t.plot(thumbnail=True)
