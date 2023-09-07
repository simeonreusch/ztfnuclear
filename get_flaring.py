#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import logging

from ztfnuclear import io
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.sample import NuclearSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.database").setLevel(logging.INFO)
logging.getLogger("ztfnuclear.sample").setLevel(logging.INFO)

# si = SampleInfo()
# meta = MetadataDB()
s = NuclearSample()

s.generate_ir_flare_sample()
s.generate_overview_pickled(flaring_only=True)
# s.generate_overview_pickled()
