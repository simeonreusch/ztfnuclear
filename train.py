#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging

from ztfnuclear.train import Model

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SEED = 10

# m = Model(
#     noisified=True,
#     noisified_test=True,
#     seed=SEED,
#     nuclear_test=False,
#     n_iter=50,
#     test_fraction=0.3,
#     train_validation_fraction=0.7,
#     grid_search_sample_size=2000,
# )
# m.train()
# m.evaluate(normalize=True)
# m.evaluate(normalize=False)

m = Model(
    noisified=True,
    noisified_test=False,
    seed=SEED,
    nuclear_test=False,
    n_iter=50,
    test_fraction=0.3,
    train_validation_fraction=0.7,
    grid_search_sample_size=2000,
)
# m.train()
m.evaluate(normalize=None)
m.evaluate(normalize="all")
m.evaluate(normalize="pred")
m.evaluate(normalize="true")
# m.classify()
