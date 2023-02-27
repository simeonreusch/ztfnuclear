#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import argparse
import json
import logging
import os
import re

from ztfnuclear.database import MetadataDB

databases = {
    "ztfnuclear": MetadataDB(),
    "bts": MetadataDB(sampletype="bts"),
    "train": MetadataDB(sampletype="train"),
}

if __name__ == "__main__":
    logger = logging.getLogger(name=__name__)
    parser = argparse.ArgumentParser(description="Import keys from json to MongoDB")

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

    json_path = commandline_args.key_name + ".json"

    databases[commandline_args.db].key_update_from_json(
        json_path=json_path, mongo_key=commandline_args.key_name
    )
    logger.info(f"Imported {json_path} to MongoDB {commandline_args.db}")
