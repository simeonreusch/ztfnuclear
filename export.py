#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re, json, argparse
from ztfnuclear.database import MetadataDB

databases = {"ztfnuclear": MetadataDB(), "bts": MetadataDB(sampletype="bts")}

if __name__ == "__main__":

    logger = logging.getLogger(name=__name__)
    parser = argparse.ArgumentParser(description="Export keys from metadata MongoDB")

    parser.add_argument(
        "key_name",
        type=str,
        help="Provide a key of the database",
    )

    parser.add_argument(
        "--db",
        type=str,
        default="ztfnuclear",
        help="Provide a name of the database",
    )

    commandline_args = parser.parse_args()

    content = databases[commandline_args.db].read_parameters(
        ["_id", commandline_args.key_name]
    )

    outfile = f"{commandline_args.key_name}.json"

    with open(outfile, "w") as fp:
        json.dump(content, fp)

    logger.info(f"Dumped from MongoDB {commandline_args.db} to {outfile}")
