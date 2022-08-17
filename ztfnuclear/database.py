#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging, collections
from tqdm import tqdm
from typing import Union, Any, Sequence, Tuple, List, Optional
from pymongo import MongoClient, UpdateOne, GEOSPHERE
from pymongo.database import Database
import pandas as pd
from healpy import ang2pix  # type: ignore
from extcats import CatalogPusher, CatalogQuery  # type: ignore

from ztfnuclear import io


class SampleInfo(object):
    """Mongo DB collection storing information about the sample"""

    def __init__(self):
        super(SampleInfo, self).__init__()
        mongo_client: MongoClient = MongoClient("localhost", 27017)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Established connection to Mongo DB")
        self.client = mongo_client
        self.db = self.client.ztfnuclear
        self.coll = self.db.info

    def update(self, data):
        """Update DB for catalog"""
        self.coll.update_one({"_id": "sample"}, {"$set": data}, upsert=True)
        self.logger.debug(f"Updated info database")


class MetadataDB(object):
    """Mongo DB collection storing all transient metadata"""

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
            has_ra = True if "RA" in testobj.keys() else False
            has_salt = True if "salt" in testobj.keys() else False
            has_peak_dates = True if "peak_dates" in testobj.keys() else False

        else:
            has_ra = False
            has_salt = False
            has_peak_dates = False

        return {
            "count": items_in_coll,
            "has_ra": has_ra,
            "has_salt": has_salt,
            "has_peak_dates": has_peak_dates,
        }


class WISE(object):
    """Interface with WISE MongoDB"""

    def __init__(self):
        super(WISE, self).__init__()

        self.logger = logging.getLogger(__name__)
        items_in_coll = self.get_statistics()
        if items_in_coll == 0:
            self.logger.warn("WISE catalogue needs to be ingested. Proceeding to do so")
            self.ingest_wise()

    def get_statistics(self):
        """
        Test the connection to the database
        """
        mongo_client: MongoClient = MongoClient("localhost", 27017)
        client = mongo_client
        db = client.allwise
        items_in_coll = db.command("collstats", "allwise")["count"]
        self.logger.info(f"Database contains {items_in_coll} entries")
        return items_in_coll

    def ingest_wise(self):
        """Ingest the WISE catalogue from the parquet file"""
        self.logger.info(
            "Reading WISE.parquet and creating Mongo database from it for querying by location. This will take a considerable amount of time, CPU and RAM!"
        )

        # build the pusher object and point it to the raw files.
        mqp = CatalogPusher.CatalogPusher(
            catalog_name="allwise",
            data_source=io.LOCALSOURCE_WISE,
            file_type="WISE.parquet",
        )

        catcols = ["RA", "Dec", "AllWISE_id"]

        mqp.assign_file_reader(
            reader_func=pd.read_parquet,
            read_chunks=False,
            columns=catcols,
            engine="fastparquet",
        )

        def _mqc_modifier(srcdict: dict):
            """
            format coordinates into geoJSON type:
            # unfortunately mongo needs the RA to be folded into -180, +180
            """
            ra = srcdict["RA"] if srcdict["RA"] < 180.0 else srcdict["RA"] - 360.0
            srcdict["pos"] = {
                "type": "Point",
                "coordinates": [ra, srcdict["Dec"]],
            }

            # add healpix index
            srcdict["hpxid_16"] = int(
                ang2pix(2**16, srcdict["RA"], srcdict["Dec"], lonlat=True, nest=True)
            )

            return srcdict

        # Modify the the source dictionary with the above defined function
        mqp.assign_dict_modifier(_mqc_modifier)

        # And now we push to Mongo
        mqp.push_to_db(
            coll_name="allwise",
            index_on=["hpxid_16", [("pos", GEOSPHERE)]],
            index_args=[{}, {}],
            overwrite_coll=True,
            append_to_coll=False,
        )

        mqp.healpix_meta(
            healpix_id_key="hpxid_16", order=16, is_indexed=True, nest=True
        )
        mqp.science_meta(
            contact="Jannis Necker, Eleni Graikou",
            email="jannis.necker@desy.de, eleni.graikou@desy.de",
            description="A great WISE sample",
            reference="http://github.com/jannisne/timewise",
        )

    def query(
        self, ra_deg: float, dec_deg: float, searchradius_arcsec: float
    ) -> Optional[dict]:
        mqc_query = CatalogQuery.CatalogQuery(
            cat_name="allwise",
            coll_name="allwise",
            ra_key="RA",
            dec_key="Dec",
            dbclient=None,
        )

        hpcp, hpcp_dist = mqc_query.findclosest(
            ra=ra_deg, dec=dec_deg, rs_arcsec=searchradius_arcsec, method="healpix"
        )
        if hpcp:
            ra = float(hpcp[0])
            dec = float(hpcp[1])
            allwise_id = float(hpcp[2])

            res = {"body": {"allwise_id": allwise_id, "dist": hpcp_dist}}

            return res

        else:
            return None
