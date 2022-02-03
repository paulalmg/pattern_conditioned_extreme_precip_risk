# This script converts the downloaded data to NetCDF and merges the control and perturbed ensemble members
# Paula Gonzalez - UoR - 2021


from subprocess import call

cmd = "grib_to_netcdf -T -o cf.nc sample_SEAsia_fcst_u_v_850_2015_12_17_cf.grib"
call(cmd,shell=True)
cmd = "grib_to_netcdf -T -o pf.nc sample_SEAsia_fcst_u_v_850_2015_12_17_pf.grib"
call(cmd,shell=True)

# Unpack files

call("ncpdq -P upk cf.nc cf.nc -O",shell=True)
call("ncpdq -P upk pf.nc pf.nc -O",shell=True)


# Define member dimension in control forecast and restructure, unpack

call("ncap2 -O -s \'defdim(\"member\",1);member[member]=0\' cf.nc -o cf.nc",shell=True)

call("ncecat -O -u member cf.nc -o cf.nc",shell=True)

call("ncpdq -O -a member,date,step,latitude,longitude cf.nc -o cf.nc",shell=True)

call("ncks -O --mk_rec_dmn member cf.nc cf.nc",shell=True)

call("ncpdq -U -O cf.nc -o cf.nc",shell=True)

    
# restructure and unpack perturbed forecasts

call("ncrename -d number,member -v number,member pf.nc",shell=True)

call("ncpdq -O -a member,date,step,latitude,longitude pf.nc -o pf.nc",shell=True)

call("ncks -O --mk_rec_dmn member pf.nc pf.nc",shell=True)

call("ncpdq -U -O pf.nc -o pf.nc",shell=True)
   
# concatenate files

cmd = "ncrcat -O cf.nc pf.nc sample_SEAsia_fcst_u_v_850_2015_12_17.nc"
call(cmd,shell=True)

# remove intermediate files

call("rm cf.nc pf.nc",shell=True)

