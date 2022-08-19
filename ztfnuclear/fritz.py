#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, json

import numpy as np
from astropy.time import Time  # type: ignore
import requests


class FritzAPI(object):
    """
    Base class for interactions with the Fritz API
    """

    def __init__(self):
        super(FritzAPI, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.base_endpoint = "https://fritz.science/"
        _token = os.getenv("FRITZ_API_TOKEN")
        if not _token:
            raise ValueError(
                "Please set 'export FRITZ_API_TOKEN=fritztokengoeshere' in your .bashrc/.zshrc"
            )
        self.logger.debug(f"Using {_token} as token for the Fritz API")
        self.headers = {"Authorization": f"token {_token}"}

    def get_transient(self, ztfid: str) -> dict:
        """
        Query the Fritz API for transient classifications
        """
        endpoint = self.base_endpoint + f"api/sources/{ztfid}"

        response = requests.request("get", endpoint, headers=self.headers)

        returndict = {}

        if response.status_code in (200, 400):
            res = response.json()
            if "data" in res.keys():
                if "classifications" in res["data"].keys():
                    classifications = res["data"]["classifications"]
                    if len(classifications) > 1:
                        classif = []
                        mod_dates = []
                        for cl in classifications:
                            classif.append(cl["classification"])
                            mod_dates.append(Time(cl["modified"], format="isot").mjd)
                        classification = classif[np.argmax(mod_dates)]
                        returndict.update({"fritz_class": classification})

                    elif len(classifications) == 1:
                        classification = classifications[0]["classification"]
                        returndict.update({"fritz_class": classification})

                if "redshift_history" in res["data"].keys():
                    if res["data"]["redshift_history"]:
                        z_hist = res["data"]["redshift_history"]
                        if len(z_hist) > 1:
                            redshifts = []
                            creation_dates = []
                            for z in z_hist:
                                redshifts.append(z["value"])
                                creation_dates.append(
                                    Time(z["set_at_utc"], format="isot").mjd
                                )
                            z = redshifts[np.argmax(creation_dates)]
                        else:
                            z = z_hist[0]["value"]
                        returndict.update({"fritz_z": z})

        return {ztfid: returndict}
