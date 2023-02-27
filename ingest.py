#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
import os
import re

import pandas as pd
from tqdm import tqdm

from ztfnuclear.baseline import baseline
from ztfnuclear.database import WISE, MetadataDB
from ztfnuclear.plot import plot_location
from ztfnuclear.sample import NuclearSample, Transient

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ztfnuclear.sample").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


s = NuclearSample()
wise = WISE()
# s.crossmatch(startindex=0)
# s.fritz(startindex=0)

# s.generate_overview_pickled()
# s.generate_overview_pickled(flaring_only=True)
