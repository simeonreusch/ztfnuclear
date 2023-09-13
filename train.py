#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
import time

from ztfnuclear.train import Model

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SEED = 10
# n_iter = 50
n_iter = 5000

start = time.time()

m = Model(
    noisified=True,
    noisified_test=False,
    seed=SEED,
    nuclear_test=False,
    n_iter=n_iter,
    test_fraction=0.3,
    train_validation_fraction=0.7,
    grid_search_sample_size=2000,
)
# m.train()
# print(m.X_test)
quit()
m.evaluate(normalize=None)
# m.evaluate(normalize="all")
# m.evaluate(normalize="pred")
# m.evaluate(normalize="true")
m.classify()

# m = Model(
#     noisified=True,
#     noisified_test=True,
#     seed=SEED,
#     nuclear_test=False,
#     n_iter=n_iter,
#     test_fraction=0.3,
#     train_validation_fraction=0.7,
#     grid_search_sample_size=2000,
# )
# # m.train()
# m.evaluate(normalize=None)
# m.evaluate(normalize="all")
# m.evaluate(normalize="pred")
# m.evaluate(normalize="true")
# # m.classify()

end = time.time()

print(f"Done with {n_iter} iterations. This took {end-start:.0f} s")
