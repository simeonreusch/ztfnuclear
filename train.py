#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
from ztfnuclear.train import Model

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


m = Model(
    noisified=True,
    noisified_validation=False,
    seed=1000,
    n_iter=50,
    validation_fraction=0.3,
    train_test_fraction=0.7,
    grid_search_sample_size=2000,
)
m.train()
m.evaluate(normalize=True)
m.evaluate(normalize=False)
