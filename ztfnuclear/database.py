#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging, collections
from typing import Union, Any, Sequence, Tuple
from pymongo import MongoClient
from pymongo.database import Database


class MetadataDB(object):
    """docstring for MetadataDB"""

    def __init__(self):
        super(MetadataDB, self).__init__()
        mongo_client: MongoClient = MongoClient("localhost", 27017)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Established connection to Mongo DB")
        self.client = mongo_client
        self.db = self.client.ztfnuclear
        self.coll = self.db.metadata

    def update(self, ztfid: str, data: dict):
        """Update DB for given ztfid"""

        self.coll.update_one({"_id": ztfid}, {"$set": data}, upsert=True)
        self.logger.debug(f"Updated database for {ztfid}")

    def read(self, ztfid: str):
        """Read entry for given ztfid"""

        entry = self.coll.find_one({"_id": ztfid})

        self.logger.debug(f"Read info for {ztfid} from database")

        return entry
