#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, collections, socket, json
from tqdm import tqdm  # type: ignore
from typing import Union, Any, Sequence, Tuple, List, Optional
from pymongo import MongoClient, UpdateOne, GEOSPHERE
from pymongo.database import Database
import pandas as pd
from healpy import ang2pix  # type: ignore
from extcats import CatalogPusher, CatalogQuery  # type: ignore
import astropy

from ztfnuclear import io

logging.getLogger("extcats.CatalogQuery").setLevel(logging.WARN)

hostname = socket.gethostname()
if hostname == "wgs33.zeuthen.desy.de":
    MONGO_PORT = 27051
else:
    MONGO_PORT = 27017

logger = logging.getLogger(__name__)


class SampleInfo(object):
    """
    Mongo DB collection storing information about the sample
    """

    def __init__(self, sampletype="nuclear"):
        super(SampleInfo, self).__init__()
        mongo_client: MongoClient = MongoClient("localhost", MONGO_PORT)
        self.logger = logging.getLogger(__name__)
        self.sampletype = sampletype
        self.logger.debug("Established connection to Mongo DB")
        self.client = mongo_client
        self.db = self.client.ztfnuclear
        if self.sampletype == "nuclear":
            self.coll = self.db.info
        elif self.sampletype == "bts":
            self.coll = self.db.info_bts
        elif self.sampletype == "train":
            self.coll = self.db.info_train

    def update(self, data):
        """
        Update DB for catalog
        """
        self.coll.update_one({"_id": "sample"}, {"$set": data}, upsert=True)
        self.logger.debug(f"Updated info database")

    def read(self):
        """
        Read from the catalog info DB
        """
        info = self.coll.find_one({"_id": "sample"})
        return info

    def read_collection(self, collection_name: str) -> list:
        """
        Read a collection of ztfids from db
        """
        info = self.read()
        if info is None:
            return None
        return info.get(collection_name)

    def ingest_ztfid_collection(self, ztfids: list, collection_name: str):
        """
        Save a list of ztfids as belonging to a selection
        """
        data = {collection_name: list(ztfids)}
        self.update(data=data)

    def ingest_selection_stats(self, collection_name: str, stats: dict):
        """
        Save a cut statistics for a collection
        """
        data = {collection_name: stats}
        self.update(data=data)


class MetadataDB(object):
    """
    Mongo DB collection storing all transient metadata
    """

    def __init__(self, sampletype="nuclear"):
        super(MetadataDB, self).__init__()
        mongo_client: MongoClient = MongoClient("localhost", MONGO_PORT)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Established connection to Mongo DB")
        self.client = mongo_client
        self.sampletype = sampletype
        self.db = self.client.ztfnuclear
        if self.sampletype == "nuclear":
            self.coll = self.db.metadata
        elif self.sampletype == "bts":
            self.coll = self.db.metadata_bts
        elif self.sampletype == "train":
            self.coll = self.db.metadata_train

    def update_transient(self, ztfid: str, data: dict):
        """
        Update DB for given ztfid
        """
        self.coll.update_one({"_id": ztfid}, {"$set": data}, upsert=True)
        self.logger.debug(f"Updated database for {ztfid}")

    def update_many(self, ztfids: list, data: List[dict]):
        """
        Update DB for a list of ztfids
        """
        bulk_operations = []
        allowed_ztfids = io.get_all_ztfids(sampletype=self.sampletype)

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

    def key_update_from_json(self, json_path: str, mongo_key: str):
        """
        Ingest the fitres json from another instance
        """
        with open(json_path) as f:
            import_dict = json.load(f)

        ztfids = import_dict["_id"]

        content = import_dict[mongo_key]
        content_full = []

        for entry in content:
            content_full.append({mongo_key: entry})

        self.update_many(ztfids=ztfids, data=content_full)
        self.logger.info(f"Updated db with key from {json_path}")

    def delete_keys(self, keys: List[str]):
        """
        Delete a key for all objects in the database
        """
        for key in keys:
            self.coll.update_many({}, {"$unset": {key: 1}})

        self.logger.info(
            f"Deleted keys {keys} for all transients in metadata db {self.sampletype}"
        )

    def read_parameters(self, params: List[str]) -> dict:
        """
        Get all values for the parameters in the list
        """
        returndict = collections.defaultdict(list)
        search_dict = {}

        for param in params:
            search_dict.update({param: 1})

        search_res = self.coll.find({}, search_dict)

        for entry in search_res:
            for param in params:
                if param in entry.keys():
                    returndict[param].append(entry[param])
                else:
                    returndict[param].append(None)

        return returndict

    def get_dataframe(
        self,
        params: List[str] = ["_id", "RA", "Dec"],
        ztfids: List[str] | None = None,
        for_training: bool = False,
    ) -> pd.DataFrame:
        """
        Get a dataframe containing all the parameters specified
        """
        search_dict = {}

        if "_id" not in params:
            params.append("_id")

        if for_training and self.sampletype in ["bts", "nuclear"]:
            params.extend(
                [
                    "crossmatch",
                    "tde_fit_exp",
                    "distnr",
                    "ZTF_bayesian",
                    "classif",
                    "ampel_z",
                    "peak_mags",
                    "salt",
                ]
            )
        elif for_training and self.sampletype == "train":
            params.extend(
                [
                    "tde_fit_exp",
                    "median_distnr",
                    "bts_peak_mag",
                    "simpleclasses",
                    "salt",
                    "ZTF_bayesian",
                    "crossmatch",
                ]
            )

        for param in params:
            search_dict.update({param: 1})

        res_dict = {}

        if ztfids is None:
            find_dict = {}
        else:
            find_dict = {"_id": {"$in": ztfids}}

        search_res = self.coll.find(find_dict, search_dict)

        from collections.abc import MutableMapping

        def flatten(d: dict, parent_key: str = "", sep: str = "_") -> dict:
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for i in search_res:
            flattened = flatten(i)
            res_dict.update({flattened.pop("_id"): flattened})

        df = pd.DataFrame.from_dict(res_dict, orient="index")

        if for_training:
            df["sample"] = self.sampletype

        if "tde_fit_exp_chisq" in df.keys():
            df["red_chisq"] = df.tde_fit_exp_chisq / df.tde_fit_exp_ndof

        if "salt_loose_bl" in df.keys():
            df["salt_red_chisq"] = df["salt_loose_bl_chisq"] / df["salt_loose_bl_ndof"]

        elif "salt" in df.keys():
            df["salt_red_chisq"] = df["salt_chisq"] / df["salt_ndof"]

        df["ztfid"] = df.index

        # if "crossmatch_TNS_type" not in df.keys():
        # df["crossmatch_TNS_type"] = None

        df.rename(
            columns={
                "tde_fit_exp_success": "success",
                "tde_fit_exp_paramdict_risetime": "rise",
                "tde_fit_exp_paramdict_decaytime": "decay",
                "tde_fit_exp_paramdict_temperature": "temp",
                "tde_fit_exp_paramdict_d_temp": "d_temp",
                "crossmatch_TNS_type": "tns_class",
                "ZTF_bayesian_bayesian_overlapping_regions_count": "overlapping_regions",
                "crossmatch_WISE_cat_Mag_W1": "wise_w1",
                "crossmatch_WISE_cat_Mag_W2": "wise_w2",
                "crossmatch_WISE_cat_Mag_W3": "wise_w3",
            },
            inplace=True,
        )

        if "wise_w1" in df.keys():
            df["wise_w1w2"] = df.wise_w1 - df.wise_w2
            df["wise_w1w2"] = df["wise_w1w2"].fillna(999)
            df["wise_w2w3"] = df.wise_w2 - df.wise_w3
            df["wise_w2w3"] = df["wise_w2w3"].fillna(999)
        # else:
        #     df["wise_w1w2"] = 999
        #     df["wise_w2w3"] = 999

        if "salt_red_chisq" in df.keys():
            df["salt_red_chisq"] = df["salt_red_chisq"].fillna(99999)

        if for_training:
            config = io.load_config()
            df.query("success == True", inplace=True)

            if self.sampletype in ["nuclear", "bts"]:
                df = df[config["train_params"]]
            else:
                df = df[config["train_params_train"]]

        return df

    def read_transient(self, ztfid: str) -> dict:
        """
        Read entry for given ztfid
        """

        entry = self.coll.find_one({"_id": ztfid})

        self.logger.debug(f"Read info for {ztfid} from database")

        return entry

    def find_by_tns(self, tns_name: str) -> dict:
        """
        Search the database for a TNS name
        """

        entry = self.coll.find_one({"TNS_name": tns_name})

        return entry

    def get_rating_overview(self, username: str = None) -> dict:
        res_dict = {}
        if username:
            res = self.coll.find(
                {f"rating.{username}": {"$exists": "true"}}, {"rating": {username: 1}}
            )
        else:
            res = self.coll.find({"rating": {"$exists": "true"}}, {"rating": 1})

        for entry in res:
            res_dict.update({entry["_id"]: entry["rating"]})

        return res_dict

    def get_statistics(self):
        """
        Get collection statistic
        """
        if self.sampletype == "nuclear":
            items_in_coll = self.db.command("collstats", "metadata")["count"]
        elif self.sampletype == "bts":
            items_in_coll = self.db.command("collstats", "metadata_bts")["count"]
        elif self.sampletype == "train":
            items_in_coll = self.db.command("collstats", "metadata_train")["count"]

        if self.sampletype in ["nuclear", "bts"]:
            testobj = self.read_transient(ztfid="ZTF19aatubsj")
        elif self.sampletype == "train":
            testobj = self.read_transient(ztfid="ZTF18abamrjl")

        if testobj:
            has_ra = True if "RA" in testobj.keys() else False
            has_salt = True if "salt" in testobj.keys() else False
            has_salt_loose_bl = True if "salt_loose_bl" in testobj.keys() else False
            has_tdefit = True if "tde_fit" in testobj.keys() else False
            has_tdefit_loose_bl = (
                True if "tde_fit_loose_bl" in testobj.keys() else False
            )
            has_peak_dates = True if "peak_dates" in testobj.keys() else False
            has_peak_mags = True if "peak_mags" in testobj.keys() else False
            has_distnr = True if "distnr" in testobj.keys() else False
            has_ampel_z = True if "ampel_z" in testobj.keys() else False
            has_wise_lc = True if "WISE_lc" in testobj.keys() else False
            has_wise_bayesian = True if "WISE_bayesian" in testobj.keys() else False
            has_wise_dust = True if "WISE_dust" in testobj.keys() else False
            has_tns = True if "TNS_name" in testobj.keys() else False
            has_ztf_bayesian = True if "ZTF_bayesian" in testobj.keys() else False

        else:
            has_ra = False
            has_salt = False
            has_salt_loose_bl = False
            has_tdefit = False
            has_tdefit_loose_bl = False
            has_peak_dates = False
            has_peak_mags = False
            has_distnr = False
            has_ampel_z = False
            has_wise_lc = False
            has_wise_bayesian = False
            has_wise_dust = False
            has_tns = False
            has_ztf_bayesian = False

        return {
            "count": items_in_coll,
            "has_ra": has_ra,
            "has_salt": has_salt,
            "has_salt_loose_bl": has_salt_loose_bl,
            "has_tdefit": has_tdefit,
            "has_tdefit_loose_bl": has_tdefit_loose_bl,
            "has_peak_dates": has_peak_dates,
            "has_peak_mags": has_peak_mags,
            "has_distnr": has_distnr,
            "has_ampel_z": has_ampel_z,
            "has_wise_lc": has_wise_lc,
            "has_wise_bayesian": has_wise_bayesian,
            "has_wise_dust": has_wise_dust,
            "has_tns": has_tns,
            "has_ztf_bayesian": has_ztf_bayesian,
        }

    def to_df(self) -> pd.DataFrame:
        """Export the metadata db as Pandas dataframe"""
        cursor = self.coll.find({})
        df = pd.DataFrame(list(cursor))
        df.rename(columns={"_id": "ztfid"}, inplace=True)
        df.set_index("ztfid", inplace=True)
        return df


class WISE(object):
    """
    Interface with WISE MongoDB
    """

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
        mongo_client: MongoClient = MongoClient("localhost", MONGO_PORT)
        client = mongo_client
        db = client.allwise
        items_in_coll = db.command("collstats", "allwise")["count"]
        self.logger.debug(f"Database contains {items_in_coll} entries")
        return items_in_coll

    def ingest_wise(self):
        """
        Ingest the WISE catalogue from the parquet file
        """
        self.logger.info(
            "Reading WISE.parquet and creating Mongo database from it for querying by location. This will take a considerable amount of time, CPU and RAM!"
        )
        if not os.path.isfile():
            df = pd.read_parquet(os.path.join(io.LOCALSOURCE_WISE, "WISE.parquet"))
            df.to_csv(os.path.join(io.LOCALSOURCE_WISE, "WISE.csv"))

        # build the pusher object and point it to the raw files.
        mqp = CatalogPusher.CatalogPusher(
            catalog_name="allwise",
            data_source=io.LOCALSOURCE_WISE,
            file_type="WISE.csv",
        )

        catcols = ["RA", "Dec", "AllWISE_id"]

        mqp.assign_file_reader(
            reader_func=pd.read_table,
            read_chunks=True,
            names=catcols,
            engine="c",
            chunksize=100000,
            sep=",",
            skiprows=1,
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
            dbclient=MongoClient("localhost", MONGO_PORT),
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
        os.remove(os.path.join(io.LOCALSOURCE_WISE, "WISE.csv"))

    def query(
        self, ra_deg: float, dec_deg: float, searchradius_arcsec: float = 3
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
            allwise_id = str(hpcp[2])

            res = {"body": {"allwise_id": allwise_id, "dist": hpcp_dist}}

            return res

        else:
            return None


class SarahAGN(object):
    """
    Interface with WISE MongoDB
    """

    def __init__(self):
        super(SarahAGN, self).__init__()

        self.logger = logging.getLogger(__name__)
        items_in_coll = self.get_statistics()
        if items_in_coll == 0:
            self.logger.warn(
                "Sarah Mechbal's AGN catalogue needs to be ingested. Proceeding to do so"
            )
            self.ingest_agncat()

    def get_statistics(self):
        """
        Test the connection to the database
        """
        mongo_client: MongoClient = MongoClient("localhost", MONGO_PORT)
        client = mongo_client
        db = client.sarah_agn
        items_in_coll = db.command("collstats", "sources")["count"]
        self.logger.debug(f"Database contains {items_in_coll} entries")
        return items_in_coll

    def ingest_agncat(self):
        """
        Ingest the WISE catalogue from the parquet file
        """
        self.logger.info("Ingesting AGN catalogue by Sarah Mechbal")

        mqp = CatalogPusher.CatalogPusher(
            catalog_name="sarah_agn",
            # data_source=hdul[1].data,
            data_source=io.LOCALSOURCE_sarah_agn,
            file_type=".fits",
        )

        mqp.assign_file_reader(
            reader_func=astropy.table.Table,
            read_chunks=False,
        )

        def _mqc_modifier(srcdict: dict):
            """
            format coordinates into geoJSON type:
            # unfortunately mongo needs the RA to be folded into -180, +180
            """
            ra = (
                srcdict["ALLW_RA"]
                if srcdict["ALLW_RA"] < 180.0
                else srcdict["ALLW_RA"] - 360.0
            )
            srcdict["pos"] = {
                "type": "Point",
                "coordinates": [ra, srcdict["ALLW_DEC"]],
            }

            # add healpix index
            srcdict["hpxid_16"] = int(
                ang2pix(
                    2**16,
                    srcdict["ALLW_RA"],
                    srcdict["ALLW_DEC"],
                    lonlat=True,
                    nest=True,
                )
            )

            return srcdict

        # Modify the the source dictionary with the above defined function
        mqp.assign_dict_modifier(_mqc_modifier)

        # And now we push to Mongo
        mqp.push_to_db(
            dbclient=MongoClient("localhost", MONGO_PORT),
            coll_name="sources",
            index_on=["hpxid_16", [("pos", GEOSPHERE)]],
            index_args=[{}, {}],
            overwrite_coll=True,
            append_to_coll=False,
            tableindex=1,
        )

        mqp.healpix_meta(
            healpix_id_key="hpxid_16", order=16, is_indexed=True, nest=True
        )
        mqp.science_meta(
            contact="Sarah Mechbal",
            email="sarah.mechbal@desy.de",
            description="A ML-reconstructed AGN catalogue",
            reference="S. Mechbal, M. Ackermann, M. Kowalski, Applications of machine learning tools in the study of AGN physical properties, A&A 2023",
        )

    def query(
        self, ra_deg: float, dec_deg: float, searchradius_arcsec: float = 3
    ) -> dict | None:
        mqc_query = CatalogQuery.CatalogQuery(
            cat_name="sarah_agn",
            coll_name="sources",
            ra_key="ALLW_RA",
            dec_key="ALLW_DEC",
            dbclient=None,
        )

        res = mqc_query.findwithin(
            ra=ra_deg, dec=dec_deg, rs_arcsec=searchradius_arcsec, method="healpix"
        )
        if res:
            df = res.to_pandas()
            df.rename(columns={"ALLW_RA": "RA", "ALLW_DEC": "Dec"}, inplace=True)
            res = df.to_dict()
        else:
            res = None

        return res
