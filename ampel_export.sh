mongoexport --collection=t2 --db=bayesian-wise-nuclear --query='{"unit": "T2BayesianBlocks"}' --out=/Users/simeon/ZTFDATA/nuclear_sample/NUCLEAR/wise_bayesian_nuclear.json --jsonArray

mongoexport --collection=t2 --db=bayesian-wise-bts --query='{"unit": "T2BayesianBlocks"}' --out=/Users/simeon/ZTFDATA/nuclear_sample/BTS/wise_bayesian_bts.json --jsonArray

mongoexport --collection=t2 --db=bayesian-wise-nuclear --query='{"unit": "T2DustEchoEval"}' --out=/Users/simeon/ZTFDATA/nuclear_sample/NUCLEAR/wise_dust_nuclear.json --jsonArray

mongoexport --collection=t2 --db=bayesian-wise-bts --query='{"unit": "T2DustEchoEval"}' --out=/Users/simeon/ZTFDATA/nuclear_sample/BTS/wise_dust_bts.json --jsonArray