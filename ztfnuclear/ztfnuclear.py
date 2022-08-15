#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

import numpy as np
import pandas as pd


class ZTFNuclear(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(self, arg):
        super(ZTFNuclear, self).__init__()
        # self.arg = arg
