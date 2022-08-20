#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re

import pandas as pd
from tqdm import tqdm

from ztfnuclear.sample import NuclearSample, Transient
from ztfnuclear.baseline import baseline
from ztfnuclear.plot import plot_location
from ztfnuclear.database import WISE, MetadataDB

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ztfnuclear.sample").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


sample = NuclearSample()
wise = WISE()
# sample.crossmatch(startindex=0)
sample.fritz(startindex=0)
