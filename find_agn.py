#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging

import pandas as pd
from pymongo import MongoClient

from ztfnuclear.ampel_api import ampel_api_cone

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.ampel_api").setLevel(logging.WARNING)

sarah_ids = []
ztf_ids = []
ras = []
decs = []
zs = []

if __name__ == "__main__":
    # client = MongoClient("localhost", 27017, maxPoolSize=50)
    # db = client.sarah_agn
    # collection = db["sources"]
    # cursor = collection.find({})

    # for document in cursor:
    #     ra = document["RA"]
    #     dec = document["DEC"]
    #     z = document["z"]
    #     ras.append(ra)
    #     decs.append(dec)
    #     zs.append(z)
    #     sarah_ids.append(document["_id"])

    # df = pd.DataFrame()
    # df["sarah_id"] = sarah_ids
    # df["ra"] = ras
    # df["dec"] = decs
    # df["z"] = zs

    # df.to_csv("sarah_agn.csv")
    df = pd.read_csv("sarah_agn.csv")

    sarah_ids_found = []
    ztf_ids_found = []
    ras = []
    decs = []
    redshifts = []
    sgscores = []
    distnrs = []

    i = 0
    k = 0

    for row in df.iterrows():
        # if i < 16452:
        #     i += 1
        #     continue
        sarah_id = row[1]["sarah_id"]
        z = row[1]["z"]
        ra = row[1]["ra"]
        dec = row[1]["dec"]
        res = ampel_api_cone(ra=ra, dec=dec, search_radius_arcsec=3)
        if len(res) > 0:
            ztfid = res[0].get("objectId")
            if ztfid is not None:
                k += 1
                sgscores.append(res[0]["candidate"]["sgscore1"])
                distnrs.append(res[0]["candidate"]["distnr"])
                ztf_ids_found.append(ztfid)
                sarah_ids_found.append(sarah_id)
                ras.append(ra)
                decs.append(dec)
                redshifts.append(z)
                print(f"found {k} ZTF transients")
                if k % 50 == 0:
                    df = pd.DataFrame()
                    df["sarah_id"] = sarah_ids_found
                    df["ztf_id"] = ztf_ids_found
                    df["ra"] = ras
                    df["dec"] = decs
                    df["z"] = redshifts
                    df["distnr"] = distnrs
                    df["sgscore"] = sgscores
                    df.to_csv(f"sarah_agn/sarah_agn_crossmatch_with_z_{k}.csv")
        print(i)
        i += 1

    df = pd.DataFrame()
    df["sarah_id"] = sarah_ids_found
    df["ztf_id"] = ztf_ids_found
    df["ra"] = ras
    df["dec"] = decs
    df["z"] = redshifts
    df["distnr"] = distnrs
    df["sgscore"] = sgscores
    df.to_csv("sarah_agn_crossmatch_with_z.csv")


# df_raw = pd.read_csv("sarah_agn.csv")
# df_raw.drop(columns=["Unnamed: 0"], inplace=True)

# print(df_raw)

# df_matched = pd.read_csv("sarah_agn_crossmatch.csv")
# df_matched.drop(columns=["Unnamed: 0"], inplace=True)

# redshifts = []

# for sarah_id in df_matched["sarah_id"].values:
#     print(sarah_id)

# print(df_raw_subset)

# for ztfid in df_matched.ztf_id.values:
#     test = df_raw.query("")

# print(df)
