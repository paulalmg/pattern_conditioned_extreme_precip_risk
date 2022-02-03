#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2015-12-02/2015-12-07/2015-12-12/2015-12-17",
    "expver": "prod",
    "levelist": "850",
    "levtype": "pl",
    "model": "glob",
    "number": "1/2/3",
    "origin": "egrr",
    "param": "131/132",
    "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480",
    "stream": "enfo",
    "time": "00:00:00",
    "type": "pf",
    "area":"35/60/-35/180", # North West South East
    "target": "sample_SEAsia_fcst_u_v_850_2015_12_17_pf.grib",
})
server = ECMWFDataServer()
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2015-12-02/2015-12-07/2015-12-12/2015-12-17",
    "expver": "prod",
    "levelist": "850",
    "levtype": "pl",
    "model": "glob",
    "origin": "egrr",
    "param": "131/132",
    "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480",
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "area":"35/60/-35/180", # North West South East
    "target": "sample_SEAsia_fcst_u_v_850_2015_12_17_cf.grib",
})


