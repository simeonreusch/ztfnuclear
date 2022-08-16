#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging, collections
from tqdm import tqdm
from typing import Union, Any, Sequence, Tuple, List
from pymongo import MongoClient, UpdateOne
from pymongo.database import Database

from ztfnuclear import io


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

    def update_transient(self, ztfid: str, data: dict):
        """Update DB for given ztfid"""

        self.coll.update_one({"_id": ztfid}, {"$set": data}, upsert=True)
        self.logger.debug(f"Updated database for {ztfid}")

    def update_many(self, ztfids: list, data: List[dict]):
        """Update DB for a list of ztfids"""

        bulk_operations = []
        allowed_ztfids = io.get_all_ztfids()

        for i, ztfid in enumerate(ztfids):
            if ztfid in allowed_ztfids:
                bulk_operations.append(
                    UpdateOne(
                        {"_id": ztfid},
                        {"$set": data[i]},
                        upsert=True,
                    )
                )
        self.coll.bulk_write(bulk_operations)
        self.logger.debug(f"Updated database for {len(ztfids)} ztfids")

    def read_parameters(self, params: List[str]):
        """Get all values for the parameters in the list"""
        returndict = collections.defaultdict(list)

        # for param in params:

        for entry in self.coll.find():
            for param in params:
                if entry.get(param, None) is not None:
                    returndict[param].append(entry[param])
                else:
                    returndict[param].append(None)

        return returndict

    def read_transient(self, ztfid: str):
        """Read entry for given ztfid"""

        entry = self.coll.find_one({"_id": ztfid})

        self.logger.debug(f"Read info for {ztfid} from database")

        return entry

    def get_statistics(self):
        """Get collection statistic"""
        items_in_coll = self.db.command("collstats", "metadata")["count"]
        testobj = self.read_transient(ztfid="ZTF19aatubsj")

        if testobj:
            if "RA" in testobj.keys():
                has_ra = True
            else:
                has_ra = False
            if "peak_dates" in testobj.keys():
                has_peak_dates = True
            else:
                has_peak_dates = False
        else:
            has_ra = False
            has_peak_dates = False

        return {
            "count": items_in_coll,
            "has_ra": has_ra,
            "has_peak_dates": has_peak_dates,
        }

    def read_all(self, property):
        """Search database for given property"""
