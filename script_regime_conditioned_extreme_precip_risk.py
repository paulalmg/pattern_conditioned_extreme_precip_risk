#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script considers an ensemble forecast for u,v @ 850 hPa over Southeast 
Asia and and derives regime-conditioned probabilities of occurrence of 
extreme rainfall, given a percentile or threshold, for both the flat and tiered
methodologies. For a more complete documentation, a README file is included.

Paula Gonzalez & Emma Howard, University of Reading, 2021

"""

# required packages

import iris
import pandas as pd
import numpy as np
import datetime as dt
from iris.experimental.equalise_cubes import equalise_attributes
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import iris.coord_categorisation
from iris.experimental import equalise_cubes
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import os

# define directories - might need changing if directory structure or file sources are changed

dir = os.path.dirname(__file__) # identifies location of package

ifile_path=os.path.join(dir, 'sample_fcst_data')
input_data_path=os.path.join(dir, 'input_data')
output_path=os.path.join(dir, 'Output')  # currently, no outputs other than figures are stored
figure_path=os.path.join(dir, 'Figures')

# to restrict to domain of interest

# large 'hemispheric' domain
cx1 = iris.Constraint(longitude=lambda x: 60<=x<=180)
cy1 = iris.Constraint(latitude=lambda y: -35<=y<=35)
# small SE Asia domain
cx2 = iris.Constraint(longitude=lambda x: 90<=x<=140)
cy2 = iris.Constraint(latitude=lambda y: -15<=y<=25)

################################################################################################################################################
################################################################################################################################################
# EXAMPLE RUN 

# Some examples set up to demonstrate the use of the method

def main():
        
    # read sample data: UKMO GloSea5 s2s fcst obtained from ECMWF - see attached sample download script
    
    u_fcst = iris.load(ifile_path + "/sample_SEAsia_fcst_u_v_850_2015_12_17.nc",'eastward_wind')
    v_fcst = iris.load(ifile_path + "/sample_SEAsia_fcst_u_v_850_2015_12_17.nc",'northward_wind')
    
    
    equalise_cubes.equalise_attributes(u_fcst)
    equalise_cubes.equalise_attributes(v_fcst)
    
    #transform from list of cube sto cube
    u_fcst=u_fcst.concatenate_cube()
    v_fcst=v_fcst.concatenate_cube()
    
    latitudes =u_fcst.coord("latitude").points
    longitudes =u_fcst.coord("longitude").points
    
    # explore dimensions - sample data has 4 starts (Base_date), 4 members, 21 lead times (time_step), 47 lats and 81 lons
    
    print(u_fcst) # member x starts x lead x lat x lon
    print(v_fcst)
    
    # read forecast start dates
    
    basetime = u_fcst.coord("Base_date").units.num2date(u_fcst.coord("Base_date").points)
    
    # obtain the valid dates combining with the lead times
    
    id_leads=u_fcst.coord("time_step").points
    
    lead=[dt.timedelta(il/24) for il in id_leads.astype(float)]  # if L is in hours, it should be timedelta(il/24), else just use il
    a,b = np.meshgrid(basetime,lead,indexing='ij')
    valid=iris.coords.AuxCoord(a+b,units=u_fcst.coord('Base_date').units,long_name='valid_time')
    
    fcst_start_dates=basetime
    fcst_valid_dates=np.array(valid.points) # array starts x leads
    
    ###############
    #  EXAMPLE 1  #
    ###############
    
    # In this example we want to consider a single valid date (target_date) forcasted at 4 different lead times (tg_leads): 0,5,10 and 15 days ahead
    
    # define target date - in this case 17-Dec-2015
    
    tg_y=2015 
    tg_m=12
    tg_d=17
    
    target_date=dt.datetime(tg_y,tg_m,tg_d)
    
    # and the target lead
    tg_lead=0
    
    # find the four corresponding forecasts
    tg_start=np.where(fcst_valid_dates[:,tg_lead]==target_date)
    
    # retain only the target forecast field
    tg_u_fcst=u_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    tg_v_fcst=v_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    
    # call regime probability assignments module
    fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs = assign_flat_tiered_regime_probabilities(tg_u_fcst,tg_v_fcst)   
    
    
    # call module that uses GPM precipitation to establish regime-conditioned probability of precipitation exceedance
    
    fcst_exceed_prob_flat,fcst_exceed_prob_tiered,latitudes,longitudes=reg_cond_exceed_probs(fcst_flat_probs,fcst_tiered_probs, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True)
       
    # call module that calculates the climatological probability of precipitation exceedance given the day of the year 
    
    clim_exceed_prob_flat,clim_exceed_prob_tiered=clim_exceed_probs(target_date, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True)
    
    
    # plot forecasted probabilities of exceedance and compared them to the climatological ones
    
    plot_fcst_and_clim_exceed_prob(target_date,tg_lead,latitudes,longitudes,fcst_exceed_prob_flat,fcst_exceed_prob_tiered,clim_exceed_prob_flat,clim_exceed_prob_tiered,kind="percentile", percentile=75, threshold=np.nan,case_label="P75 Exceedance probabilities",fig_label="P75_exceed_probs")
        
    
    # test the exceedance of observed precip - in this case as downloaded from GPM
    
    obs_pr_file="3B-DAY.MS.MRG.3IMERG.%04d%02d%02d-S000000-E235959.V06.nc4"%(target_date.year,target_date.month,target_date.day)   
    
    obs_precip,obs_exceed=test_gpm_exceed(target_date, obs_precip_file=obs_pr_file, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True)
    
        
    # plot observed precip and exceedance
    
    obs_exceed=obs_exceed.astype(float)
    obs_exceed[obs_exceed==0]=np.nan
    plot_obs_precip_and_exceed(target_date,latitudes,longitudes,obs_precip,obs_exceed,case_label="P75 Exceedance",fig_label="P75_exceed")
    
    # plot the contribution of the relevant regimes
    
    plot_relevant_regimes(target_date,tg_lead,fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True, case_label="P75 Exceedance",fig_label="P75_exceed")

    
    ###############
    #  EXAMPLE 2  #
    ###############
    
    
    # now we try looking into the probabilities for other lead time - 5 days ahead
    
    tg_lead=5
    
    # find the four corresponding forecasts
    tg_start=np.where(fcst_valid_dates[:,tg_lead]==target_date)
    
    
    # retain only the target forecast field
    tg_u_fcst=u_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    tg_v_fcst=v_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    
    # call regime probability assignments module
    
    fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs = assign_flat_tiered_regime_probabilities(tg_u_fcst,tg_v_fcst)   
    
    # call module that uses GPM precipitation to establish regime-conditioned probability of precipitation exceedance
    
    fcst_exceed_prob_flat,fcst_exceed_prob_tiered,latitudes,longitudes=reg_cond_exceed_probs(fcst_flat_probs,fcst_tiered_probs, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True)
       
    # call module that calculates the climatological probability of precipitation exceedance given the day of the year 
    
    clim_exceed_prob_flat,clim_exceed_prob_tiered=clim_exceed_probs(target_date, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True)
    
    ####
    # plot forecasted probabilities of exceedance and compared them to the climatological ones
    
    plot_fcst_and_clim_exceed_prob(target_date,tg_lead,latitudes,longitudes,fcst_exceed_prob_flat,fcst_exceed_prob_tiered,clim_exceed_prob_flat,clim_exceed_prob_tiered,kind="percentile", percentile=75, threshold=np.nan,case_label="P75 Exceedance probabilities",fig_label="P75_exceed_probs")
   
    # plot the contribution of the relevant regimes
    
    plot_relevant_regimes(target_date,tg_lead,fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs, kind="percentile", percentile=75, threshold=np.nan, window=60, res="1p5degree",  seas=True, case_label="P75 Exceedance",fig_label="P75_exceed")
  


     
    ###############
    #  EXAMPLE 3  #
    ###############   
    
    
    ### Now this tests the use of threshold exceedance
    
    # and the target lead
    tg_lead=0
    
    # find the four corresponding forecasts
    tg_start=np.where(fcst_valid_dates[:,tg_lead]==target_date)
    
    
    # retain only the target forecast field
    tg_u_fcst=u_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    tg_v_fcst=v_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    
    # call regime probability assignments module
    
    fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs = assign_flat_tiered_regime_probabilities(tg_u_fcst,tg_v_fcst)   
    
    # call module that uses GPM precipitation to establish regime-conditioned probability of precipitation exceedance
    
    fcst_exceed_prob_flat,fcst_exceed_prob_tiered,latitudes,longitudes=reg_cond_exceed_probs(fcst_flat_probs,fcst_tiered_probs, kind="threshold", percentile=np.nan, threshold=25, window=60, res="1p5degree",  seas=False)
       
    # call module that calculates the climatological probability of precipitation exceedance given the day of the year 
    
    clim_exceed_prob_flat,clim_exceed_prob_tiered=clim_exceed_probs(target_date, kind="threshold", percentile=np.nan, threshold=25, res="1p5degree",  seas=False)
    
    
    ####
    # plot forecasted probabilities of exceedance and compared them to the climatological ones
    
    plot_fcst_and_clim_exceed_prob(target_date,tg_lead,latitudes,longitudes,fcst_exceed_prob_flat,fcst_exceed_prob_tiered,clim_exceed_prob_flat,clim_exceed_prob_tiered,kind="threshold", percentile=np.nan, threshold=25,case_label="25 mm/day Exceedance probabilities",fig_label="25mm_exceed_probs")
        
    
    # test the exceedance of observed precip - in this case as downloaded from GPM
    
    obs_pr_file="3B-DAY.MS.MRG.3IMERG.%04d%02d%02d-S000000-E235959.V06.nc4"%(target_date.year,target_date.month,target_date.day)   
    
    obs_precip,obs_exceed=test_gpm_exceed(target_date, obs_precip_file=obs_pr_file, kind="threshold", percentile=np.nan, threshold=25, res="1p5degree",  seas=False)
    
        
    # plot observed precip and exceedance
    
    obs_exceed=obs_exceed.astype(float)
    obs_exceed[obs_exceed==0]=np.nan
    plot_obs_precip_and_exceed(target_date,latitudes,longitudes,obs_precip,obs_exceed,case_label="25 mm/day Exceedance",fig_label="25mm_exceed")
    
    
    # plot the contribution of the relevant regimes
    
    plot_relevant_regimes(target_date,tg_lead,fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs, kind="threshold", percentile=np.nan, threshold=25, res="1p5degree",  seas=False, case_label="25 mm/day Exceedance",fig_label="25mm_exceed")

    
    
    ####
    # now we try looking into the probabilities for other lead time - 10 days ahead
    
    tg_lead=10
    
    # find the four corresponding forecasts
    tg_start=np.where(fcst_valid_dates[:,tg_lead]==target_date)
    
    
    # retain only the target forecast field
    tg_u_fcst=u_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    tg_v_fcst=v_fcst[:,tg_start[0],tg_lead,:,:] # members x lats x lons
    
    # call regime probability assignments module
    
    fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs = assign_flat_tiered_regime_probabilities(tg_u_fcst,tg_v_fcst)   
    
    # call module that uses GPM precipitation to establish regime-conditioned probability of precipitation exceedance
    
    fcst_exceed_prob_flat,fcst_exceed_prob_tiered,latitudes,longitudes=reg_cond_exceed_probs(fcst_flat_probs,fcst_tiered_probs, kind="threshold", percentile=np.nan, threshold=25, window=60, res="1p5degree",  seas=False)
       
    # call module that calculates the climatological probability of precipitation exceedance given the day of the year 
    
    clim_exceed_prob_flat,clim_exceed_prob_tiered=clim_exceed_probs(target_date, kind="threshold", percentile=np.nan, threshold=25, window=60, res="1p5degree",  seas=False)
    
    
    ####
    # plot forecasted probabilities of exceedance and compared them to the climatological ones
    
    plot_fcst_and_clim_exceed_prob(target_date,tg_lead,latitudes,longitudes,fcst_exceed_prob_flat,fcst_exceed_prob_tiered,clim_exceed_prob_flat,clim_exceed_prob_tiered,kind="threshold", percentile=np.nan, threshold=25,case_label="25 mm/day Exceedance",fig_label="25mm_exceed")
          
    # plot the contribution of the relevant regimes
    
    plot_relevant_regimes(target_date,tg_lead,fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs, kind="threshold", percentile=np.nan, threshold=25, res="1p5degree",  seas=False, case_label="25 mm/day Exceedance",fig_label="25mm_exceed")

    return


# -------------------------------------------------------------------------------------------------------------------------

###############
#   MODULES   #
###############

# start of the package module definitions - to test the code, run main()

###
## MODULE: regime assignments

def assign_flat_tiered_regime_probabilities(u_in,v_in):
# this module assigns flat clusters to u,v forecast data with structure (member,lat,lon)
  
    ncats=51 # this is fixed fore the flat/tiered methodologies
    
# restrict to 'small' SE Asia domain    
    u2 = u_in.extract(cx2&cy2)
    v2 = v_in.extract(cx2&cy2)
    
#####
# flat regimes assignment    
    
# flat centroids   
    flat_centroids = iris.load(input_data_path+"/ERA5_wind_centroids_flat.nc")
    uc_fl =flat_centroids.extract("eastward_wind" )[0].regrid(u2,iris.analysis.Linear())
    vc_fl =flat_centroids.extract("northward_wind")[0].regrid(u2,iris.analysis.Linear())
  
# assign according to minimized Euclidian distance
    nr,flat_reg = [], []
    for j in range(u_in.shape[0]): # these are the ensemble members
          a = ((uc_fl.data-u2[j].data)**2 + (vc_fl.data-v2[j].data)**2).mean(axis=(1,2))
          nr.append(np.argmin(a)+1)
    flat_reg=np.array(nr)  # most likely flat regime for each ensemble member
     
# calculate probabilistic prediction from flat forecast ensemble
    nmems=len(nr) # number of ensemble members
    fcst_flat_prob=np.zeros(ncats)
    for kc in range(1,ncats+1):
        fcst_flat_prob[kc-1]=np.sum(flat_reg==kc)/nmems
    
#####
## tiered regimes assignment

# restrict to 'large' hemispheric domain for Tier 1 regime assigment        
    u1 = u_in.extract(cx1&cy1)
    v1 = v_in.extract(cx1&cy1)  
    
# Tier 1 centroids      
    t1_centroids = iris.load(input_data_path+"/ERA5_wind_centroids_tier1.nc")
    uc_t1 =t1_centroids.extract("eastward_wind" )[0].regrid(u1,iris.analysis.Linear())
    vc_t1 =t1_centroids.extract("northward_wind")[0].regrid(u1,iris.analysis.Linear())

# Tier 2 centroids
    t2_centroids = iris.load(input_data_path+"/ERA5_wind_centroids_tier2.nc")
    uc_t2 =t2_centroids.extract("eastward_wind" )[0].regrid(u2,iris.analysis.Linear())
    vc_t2 =t2_centroids.extract("northward_wind")[0].regrid(u2,iris.analysis.Linear())
    uc_t2.coord('tier 1').rename("tier_1")
    vc_t2.coord('tier 1').rename("tier_1")
    
# t2 regimes per each t1
    nt2=[5,7,7,8,6,5,7,6]   
    
# Assigns first a Tier 1 regime and then a Tier 2 
    nr1,nr2,t2_reg = [],[],[]
    for j in range(u_in.shape[0]): # this is for each ensemble member
       a1 = ((uc_t1.data-u1[j].data)**2 + (vc_t1.data-v1[j].data)**2).mean(axis=(1,2))
       t1 = np.argmin(a1)+1
       ct=iris.Constraint(tier_1=t1)
       a2 = ((uc_t2.extract(ct).data-u2[j].data)**2 + (vc_t2.extract(ct).data-v2[j].data)**2).mean(axis=(1,2))
       nr1.append(np.argmin(a1)+1)
       nr2.append(np.argmin(a2)+1)
       
# switch to linear 1-51 assignment   
    for m in range(0,nmems):   
        ii=0
        if nr1[m]>1:
           ii=np.sum(nt2[0:(nr1[m]-1)])
        ii=ii+nr2[m]
        t2_reg.append(ii) # most likely tiered regime for each ensemble member - identified from 1-51
       
# calculate probabilistic prediction from tiered forecast ensemble
    fcst_tiered_prob=np.zeros(ncats)
    for kc in range(1,ncats+1):
        fcst_tiered_prob[kc-1]=np.sum(np.array(t2_reg)==kc)/nmems

    return fcst_flat_prob,fcst_tiered_prob,flat_reg,t2_reg

###
# MODULE: regime-conditioned GPM probabilities of exceedance
def reg_cond_exceed_probs(prob_reg_flat,prob_reg_tiered, kind="percentile", percentile=np.nan, threshold=np.nan, window=np.nan, res="1p5degree",  seas=True):
    
    # assess possible errors and misisng combinations
    
    if kind == "threshold" and seas:
        exit("absolute threshold cannot be combined with seasonality")
    
    if seas==False and not kind=="threshold":
        exit("If a seasonally dependent percentile climatology shouldn't be considered, re-run the precip exceedance provided code.")

    if res!="1p5degree":
        exit("If a resolution different than S2S' 1.5 degree should be considered, re-run the precip exceedance provided code.")
    
    # read GPM probability of precip exceedance conditioned by the regimes
    
    if kind=="percentile":
        
        assert(not(np.isnan(percentile)))
        assert(not(np.isnan(window)))
        
        if percentile in [75,90]:
            
            if window in [60]:

                data_flat=iris.load(input_data_path + "/GPM_flat_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_flat=data_flat.extract("P%d exceedence"%(percentile))[0]

                data_tiered=iris.load(input_data_path + "/GPM_tier2_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_tiered=data_tiered.extract("P%d exceedence"%(percentile))[0]
                
                latitudes= data_flat.extract("P%d exceedence"%(percentile))[0].coord('latitude')
                longitudes= data_flat.extract("P%d exceedence"%(percentile))[0].coord('longitude')
                
            else:
                
                exit("For seasonality windows different than 60 days, re-run the precip exceedance provided code.")
            
        else:
            
          exit("For percentiles different than P75 or P90, re-run the precip exceedance provided code.")      
        
    elif kind=="threshold":
        
        assert(not(np.isnan(threshold)))
        
        if threshold in [25]:

            data_flat=iris.load(input_data_path + "/GPM_flat_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_flat=data_flat.extract("%d mm/day exceedence"%(threshold))[0]

            data_tiered=iris.load(input_data_path + "/GPM_tier2_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_tiered=data_tiered.extract("%d mm/day exceedence"%(threshold))[0]  
            
            
            latitudes= data_flat.extract("%d mm/day exceedence"%(threshold))[0].coord('latitude')
            longitudes= data_flat.extract("%d mm/day exceedence"%(threshold))[0].coord('longitude')
            
        else:
            
           exit("For thresholds different than 25mm, re-run the precip exceedance provided code.")  
    
        
  # conditional precipitation exceedance hindcast
    ncats=51 # pre-determined for both flat and tiered regimes      
    fcst_exceed_prob_flat=0
    fcst_exceed_prob_tiered=0
    for ki in range(0,ncats):
        fcst_exceed_prob_flat += np.multiply(prob_reg_flat[ki,None,None],cond_prob_flat.data[ki,:,:])
        fcst_exceed_prob_tiered += np.multiply(prob_reg_tiered[ki,None,None],cond_prob_tiered.data[ki,:,:])       
        
   # output latitudes and longitudes in case the fcst had to be interpolated
   
    latitudes=latitudes.points
    longitudes=longitudes.points

        
    return fcst_exceed_prob_flat,fcst_exceed_prob_tiered,latitudes,longitudes

###
# MODULE: climatological probabilities of exceedance given the day of year and the corresponding probabilites of occurrence of each regime
    
def clim_exceed_probs(date, kind="percentile", percentile=np.nan, threshold=np.nan, window=np.nan, res="1p5degree",  seas=True):
    
    # load climatological probabilites of occurrence of the regimes as function of doyr (1:366)
    
    clim_prob_regimes_tiered=pd.read_csv(input_data_path+"/ERA5_tier2_regimes_climatological_probability.csv",index_col=0)
    clim_prob_regimes_flat=pd.read_csv(input_data_path+"/ERA5_flat_regimes_climatological_probability.csv",index_col=0)  

    # determine day of the year
    
    doyr=(pd.to_datetime(date).replace(year=2000)-dt.datetime(2000,1,1)).days 
    
    # restrict probabilities to that day
    
    prob_reg_flat=np.array(clim_prob_regimes_flat)[:,doyr]
    prob_reg_tiered=np.array(clim_prob_regimes_tiered)[:,doyr]
    
    # assess possible errors and misisng combinations
    
    if kind == "threshold" and seas:
        exit("absolute threshold cannot be combined with seasonality")
    
    if seas==False and not kind=="threshold":
        exit("If a seasonally dependent climatology shouldn't be considered, re-run the precip exceedance provided code.")

    if res!="1p5degree":
        exit("If a resolution different than S2S' 1.5 degree should be considered, re-run the precip exceedance provided code.")
    
    # read GPM probability of precip exceedance conditioned by the regimes
    
    if kind=="percentile":
        
        assert(not(np.isnan(percentile)))
        assert(not(np.isnan(window)))
        
        if percentile in [75,90]:
            
            if window in [60]:

                data_flat=iris.load(input_data_path + "/GPM_flat_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_flat=data_flat.extract("P%d exceedence"%(percentile))[0]

                data_tiered=iris.load(input_data_path + "/GPM_tier2_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_tiered=data_tiered.extract("P%d exceedence"%(percentile))[0]
                
            else:
                
                exit("For seasonality windows different than 60 days, re-run the precip exceedance provided code.")
            
        else:
            
          exit("For percentiles different than P75 or P90, re-run the precip exceedance provided code.")      
        
    elif kind=="threshold":
        
        assert(not(np.isnan(threshold)))
        
        if threshold in [25]:

            data_flat=iris.load(input_data_path + "/GPM_flat_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_flat=data_flat.extract("%d mm/day exceedence"%(threshold))[0]

            data_tiered=iris.load(input_data_path + "/GPM_tier2_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_tiered=data_tiered.extract("%d mm/day exceedence"%(threshold))[0]  
            
        else:
            
           exit("For thresholds different than 25mm, re-run the precip exceedance provided code.")  
    
    
    
        
  # conditional precipitation exceedance hindcast
    ncats=51 # pre-determined for both flat and tiered regimes      
    clim_exceed_prob_flat=0
    clim_exceed_prob_tiered=0
    for ki in range(0,ncats):
        clim_exceed_prob_flat += np.multiply(prob_reg_flat[ki,None,None],cond_prob_flat.data[ki,:,:])
        clim_exceed_prob_tiered += np.multiply(prob_reg_tiered[ki,None,None],cond_prob_tiered.data[ki,:,:])        
        
    return clim_exceed_prob_flat,clim_exceed_prob_tiered


 
####
# MODULE: test exceedance in the observed GPM data

def test_gpm_exceed(date, obs_precip_file, kind="percentile", percentile=np.nan, threshold=np.nan, window=np.nan, res="1p5degree",  seas=True):
    # assess possible errors and misisng combinations
    
    if kind == "threshold" and seas:
        exit("absolute threshold cannot be combined with seasonality")
    
    if seas==False and not kind=="threshold":
        exit("If a seasonally dependent percentile climatology shouldn't be considered, re-run the precip exceedance provided code.")

    if res!="1p5degree":
        exit("If a resolution different than S2S' 1.5 degree should be considered, re-run the precip exceedance provided code.")
        
    # read GPM threshold for the exceedance
    
    if kind=="percentile":
        
        assert(not(np.isnan(percentile)))
        assert(not(np.isnan(window)))
        
        if percentile in [75,90]:
            
            if window in [60]:

                clim=iris.load(input_data_path + "/GPM_clim_%d_day_%d_percentile_%s.nc"%(window,percentile,res))[0]
                
            else:
                
                exit("For seasonality windows different than 60 days, re-run the precip exceedance provided code.")
            
        else:
            
          exit("For percentiles different than P75 or P90, re-run the precip exceedance provided code.")      
        
    elif kind=="threshold":
        
        assert(not(np.isnan(threshold)))
        
        if threshold in [25]:

            clim=iris.load(input_data_path + "/GPM_clim_30_day_mean_%s.nc"%(res))[0] # we just need this for the grid description
            
        else:
            
           exit("For thresholds different than 25mm, re-run the precip exceedance provided code.")  
        
     # read observed precip and match domain
     
    x0,x1=(90,140)
    y0,y1=(-15,25)
 
    cx = iris.Constraint(longitude=lambda x:x0<=x<=x1)
    cy = iris.Constraint(latitude =lambda y:y0<=y<=y1)
    
    obs_precip=iris.load(input_data_path + "/" + obs_precip_file,cx&cy).extract("Daily accumulated precipitation (combined microwave-IR) estimate") 
    equalise_attributes(obs_precip)
    obs_precip=obs_precip.concatenate_cube()
    obs_precip.transpose([0,2,1])
    
    # reg-grid to climatology resolution
    
    obs_precip=obs_precip.regrid(clim,iris.analysis.Linear())
    
    # output the exceedance
    
    obs_exceed = iris.cube.CubeList()
    
    # test against the climatology or threshold
    
    clim_t = clim.extract(iris.Constraint(time = lambda tt: tt.point.day==date.day and tt.point.month==date.month))
    if kind=="threshold":
      obs_exceed.append(obs_precip-threshold)
    else:
      obs_exceed.append(obs_precip-clim_t)
    obs_exceed=obs_exceed.merge_cube()
    obs_exceed.data=(obs_exceed.data>0).astype(int)
    
    # return as numpy arrays

    obs_precip=np.squeeze(np.array(obs_precip.data)) 
    obs_exceed=np.squeeze(np.array(obs_exceed.data))   
  
    return obs_precip,obs_exceed   

####
# MODULE: plot forecasted probabilities of exceedance together with climatological probabilities

def plot_fcst_and_clim_exceed_prob(datet,lead,latitudes,longitudes,cond_pr_flat,cond_pr_tiered,cond_pr_flat_clim,cond_pr_tiered_clim,kind="threshold", percentile=np.nan, threshold=np.nan, case_label="undefined exceedance case",fig_label="undefined_exceedance_case"):
   
    # determines appropriate color scales - this might need to change for other exceedance limits
    
    if kind=='percentile' and percentile==75 :
    
        clevs = np.array([0,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)
    
    elif kind=='percentile' and percentile==90:
        
        clevs = np.array([0,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)

    else:  
        
        clevs = np.array([0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)    

    # create figure
    
    fig=plt.figure(figsize=(9,8))
         
    ax=plt.subplot(221,projection=ccrs.PlateCarree())
    plt.title("conditional prob flat [%]")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    a=plt.contourf(longitudes,latitudes,np.squeeze(cond_pr_flat)*100,clevs,axes=ax,cmap=cmap, norm=norm )
    ax.coastlines()
    
    ax=plt.subplot(222,projection=ccrs.PlateCarree())
    plt.title("conditional prob tiered [%]")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    a=plt.contourf(longitudes,latitudes,np.squeeze(cond_pr_tiered)*100,clevs,axes=ax,cmap=cmap, norm=norm )
    ax.coastlines()
    
    ax=plt.subplot(223,projection=ccrs.PlateCarree())
    plt.title("climatological prob flat [%]")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    a=plt.contourf(longitudes,latitudes,np.squeeze(cond_pr_flat_clim)*100,clevs,axes=ax,cmap=cmap, norm=norm )
    ax.coastlines()
 
    ax=plt.subplot(224,projection=ccrs.PlateCarree())
    plt.title("climatological prob tiered [%]")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    a=plt.contourf(longitudes,latitudes,np.squeeze(cond_pr_tiered_clim)*100,clevs,axes=ax,cmap=cmap, norm=norm )
    ax.coastlines()
    
    
    fig.subplots_adjust(top=0.94,bottom=0.025,left=0.015,right=0.85,hspace=0.3,wspace=0.04)
    cax=fig.add_axes([0.88,0.11,0.05,0.73])
    fig.colorbar(a,cax=cax)
    
    fig.suptitle(case_label + " - Target date:  %04d-%02d-%02d - Fcst lead: %d"%(datet.year,datet.month,datet.day,lead),y=0.99,fontsize='x-large',fontweight='bold',x=0.45)
    plt.savefig(figure_path + "/" + fig_label + "_%04d-%02d-%02d_lead_%d.png"%(datet.year,datet.month,datet.day,lead))
    plt.show()
    
    return

####
# MODULE: plot forecasted probabilities of exceedance together with climatological probabilities

def plot_obs_precip_and_exceed(datet,latitudes,longitudes,obs_pr,obs_exceed,case_label="undefined exceedance case",fig_label="undefined_exceedance_case"):
    
    # define a suitable color scale and boundaries
    
    clevs_pr = np.array([0,.1,.5,1,2,4,7,15,30,60,120])
    cmap_data =[(1,1,1)]+ [get_cmap("YlGnBu")(x) for x in np.linspace(0,1,len(clevs_pr-1))]
    cmap_pr = mcolors.ListedColormap(cmap_data, 'precipitation')
    norm_pr = mcolors.BoundaryNorm(clevs_pr, cmap_pr.N)

    # create figure
    
    fig=plt.figure(figsize=(6,4))
         
    ax=plt.subplot(121,projection=ccrs.PlateCarree())
    plt.title("Observed exceedance")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    a=plt.pcolormesh(longitudes,latitudes,obs_exceed,vmin=0,vmax=2)
    ax.coastlines()
    
    ax=plt.subplot(122,projection=ccrs.PlateCarree())
    plt.title("Precipitation [mm/day]")
    plt.xlim(longitudes[0],longitudes[-1])
    plt.ylim(latitudes[-1],latitudes[0])
    b=plt.contourf(longitudes,latitudes,obs_pr, clevs_pr,axes=ax,cmap=cmap_pr, norm=norm_pr)
    ax.coastlines()
            
    
    fig.subplots_adjust(top=0.9,bottom=0.025,left=0.05,right=0.95,hspace=0.3,wspace=0.04)
    cax=fig.add_axes([0.15,0.11,0.72,0.05])
    fig.colorbar(b,cax=cax,orientation='horizontal')
    
    fig.suptitle(case_label + " - Date:  %04d-%02d-%02d"%(datet.year,datet.month,datet.day),y=0.97,fontsize='x-large',fontweight='bold',x=0.5)
    plt.savefig(figure_path + "/Obs_precip_" + fig_label + "_%04d-%02d-%02d.png"%(datet.year,datet.month,datet.day))
    plt.show()
    
    return
        

#######
## MODULE: plots the centroid and exceedance of all the relevant contributing regimes alogn with their forecaste probabilities    
    

def plot_relevant_regimes(datet,lead,fcst_flat_probs,fcst_tiered_probs,fcst_flat_regs,fcst_tiered_regs, kind="percentile", percentile=np.nan, threshold=np.nan, window=np.nan, res="1p5degree",  seas=True, case_label="undefined exceedance case",fig_label="undefined_exceedance_case"):
    
    # determine number of contributing regimes for each case
    nr_flat=np.sum(fcst_flat_probs>0)
    nr_tiered=np.sum(fcst_tiered_probs>0)
    
    # retain unique regimes   
    fcst_flat_regs=np.unique(fcst_flat_regs)
    fcst_tiered_regs=np.unique(fcst_tiered_regs)
    
    
##### identify the relevant centroids in u,v for the SE Asia domain
    
# flat centroids  
    
    flat_centroids = iris.load(input_data_path+"/ERA5_wind_centroids_flat.nc")
    uc_fl =flat_centroids.extract("eastward_wind" )[0].extract(cx2&cy2)
    vc_fl =flat_centroids.extract("northward_wind")[0].extract(cx2&cy2) 
    
    # retain relevant flat centroids
    
    rel_uc_flat=[]
    rel_vc_flat=[]
    rel_prob_flat=[]
    for kr in range(0,nr_flat):
        rel_uc_flat.append(uc_fl[(fcst_flat_regs[kr]-1)].data)
        rel_vc_flat.append(vc_fl[(fcst_flat_regs[kr]-1)].data)
        rel_prob_flat.append(fcst_flat_probs[(fcst_flat_regs[kr]-1)])
        

# Tier 2 centroids
        
    t2_centroids = iris.load(input_data_path+"/ERA5_wind_centroids_tier2.nc")
    uc_t2 =t2_centroids.extract("eastward_wind" )[0].extract(cx2&cy2)
    vc_t2 =t2_centroids.extract("northward_wind")[0].extract(cx2&cy2)
    
    # retain relevant tiered centroids
    
    rel_uc_tiered=[]
    rel_vc_tiered=[]
    rel_prob_tiered=[]
    for kr in range(0,nr_tiered):
        rel_uc_tiered.append(uc_t2[(fcst_tiered_regs[kr]-1)].data)
        rel_vc_tiered.append(vc_t2[(fcst_tiered_regs[kr]-1)].data)
        rel_prob_tiered.append(fcst_tiered_probs[(fcst_tiered_regs[kr]-1)])
    
# retain coordinates for plots
        
    lats_cen=uc_t2.coord("latitude").points 
    lons_cen =uc_t2.coord("longitude").points  
    
    
    ### sort regimes according to probabilities

    flat_ind=np.array(rel_prob_flat).argsort()[::-1] # descending order
    tiered_ind=np.array(rel_prob_tiered).argsort()[::-1]
    
    # sort probabilities and regimes
    
    rel_prob_flat=list(np.array(rel_prob_flat)[flat_ind])
    rel_prob_tiered=list(np.array(rel_prob_tiered)[tiered_ind])
    
    fcst_flat_regs=list(np.array(fcst_flat_regs)[flat_ind])
    fcst_tiered_regs=list(np.array(fcst_tiered_regs)[tiered_ind])
    
    rel_uc_flat=list(np.array(rel_uc_flat)[flat_ind])
    rel_vc_flat=list(np.array(rel_vc_flat)[flat_ind])
    
    rel_uc_tiered=list(np.array(rel_uc_tiered)[tiered_ind])
    rel_vc_tiered=list(np.array(rel_vc_tiered)[tiered_ind])

#########  retain the exceedance probabilities associated with each relevant regime
    
        # assess possible errors and misisng combinations
    
    if kind == "threshold" and seas:
        exit("absolute threshold cannot be combined with seasonality")
    
    if seas==False and not kind=="threshold":
        exit("If a seasonally dependent percentile climatology shouldn't be considered, re-run the precip exceedance provided code.")

    if res!="1p5degree":
        exit("If a resolution different than S2S' 1.5 degree should be considered, re-run the precip exceedance provided code.")
        
# read GPM probability of precip exceedance conditioned by the regimes
    
    if kind=="percentile":
        
        assert(not(np.isnan(percentile)))
        assert(not(np.isnan(window)))
        
        if percentile in [75,90]:
            
            if window in [60]:

                data_flat=iris.load(input_data_path + "/GPM_flat_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_flat=data_flat.extract("P%d exceedence"%(percentile))[0]

                data_tiered=iris.load(input_data_path + "/GPM_tier2_clim_%d_day_exceed_P%d_%s_seas.nc"%(window,percentile,res))
                cond_prob_tiered=data_tiered.extract("P%d exceedence"%(percentile))[0]
                
                latitudes= data_flat.extract("P%d exceedence"%(percentile))[0].coord('latitude')
                longitudes= data_flat.extract("P%d exceedence"%(percentile))[0].coord('longitude')
                
            else:
                
                exit("For seasonality windows different than 60 days, re-run the precip exceedance provided code.")
            
        else:
            
          exit("For percentiles different than P75 or P90, re-run the precip exceedance provided code.")      
        
    elif kind=="threshold":
        
        assert(not(np.isnan(threshold)))
        
        if threshold in [25]:

            data_flat=iris.load(input_data_path + "/GPM_flat_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_flat=data_flat.extract("%d mm/day exceedence"%(threshold))[0]

            data_tiered=iris.load(input_data_path + "/GPM_tier2_threshold_exceed_%d_mm_%s.nc"%(threshold,res))
            cond_prob_tiered=data_tiered.extract("%d mm/day exceedence"%(threshold))[0]  
            
            
            latitudes= data_flat.extract("%d mm/day exceedence"%(threshold))[0].coord('latitude')
            longitudes= data_flat.extract("%d mm/day exceedence"%(threshold))[0].coord('longitude')
            
        else:
            
           exit("For thresholds different than 25mm, re-run the precip exceedance provided code.")  
    
   
    latitudes=latitudes.points
    longitudes=longitudes.points
    
    # retain relevant flat exceedance
    
    rel_exc_flat=[]
    for kr in range(0,nr_flat):
        rel_exc_flat.append(cond_prob_flat[(fcst_flat_regs[kr]-1)].data)
   
    # retain relevant tiered exceedance
    
    rel_exc_tiered=[]  
    for kr in range(0,nr_tiered):
        rel_exc_tiered.append(cond_prob_tiered[(fcst_tiered_regs[kr]-1)].data)
        

    
    
    
###### Plotting 
        
     # determines appropriate color scales - this might need to change for other exceedance limits
    
    if kind=='percentile' and percentile==75 :
    
        clevs = np.array([0,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)
    
    elif kind=='percentile' and percentile==90:
        
        clevs = np.array([0,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)

    else:  
        
        clevs = np.array([0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
        cmap_data =[(1,1,1)]+ [get_cmap("YlOrRd")(x) for x in np.linspace(0,1,len(clevs-1))]
        cmap = mcolors.ListedColormap(cmap_data, 'probability')
        norm = mcolors.BoundaryNorm(clevs, cmap.N)    
        
        
        
 #### flat

    fig=plt.figure()
    
    # determine best row/column structure
    
    nrows=1
    ncols=1
    
    if nr_flat>2:
        nrows=2
        ncols=round(nr_flat/nrows)
        if ncols>4:
           nrows=3
           ncols=round(nr_flat/nrows) 
    if nr_flat>6:
        nrows=3
        ncols=round(nr_flat/nrows)
        if ncols>4:
           nrows=4
           ncols=round(nr_flat/nrows)
    if nr_flat>9:
        nrows=4
        ncols=round(nr_flat/nrows)
        if ncols>4:
           nrows=5
           ncols=round(nr_flat/nrows)

    
    
    for ks in range(nr_flat):

  # add every single subplot to the figure with a for loop

      # exceedance
      ax = fig.add_subplot(nrows,ncols,ks+1,projection=ccrs.PlateCarree())    
      plt.xlim(longitudes[0],longitudes[-1])
      plt.ylim(latitudes[-1],latitudes[0])     
      plt.title("Flat " + str(fcst_flat_regs[ks]) + " - prob [%]: "  + str(int(rel_prob_flat[ks]*100)))
      a=plt.contourf(longitudes,latitudes,np.array(rel_exc_flat[ks])*100,clevs,axes=ax,cmap=cmap, norm=norm)     
      ax.coastlines()
      # centroid winds
      k=12
      #q=plt.quiver(era5.extract('eastward_wind')[0][::k,::k],era5.extract('northward_wind')[0][::k,::k],pivot="mid",headwidth=3,width=0.01,scale=120,axes=ax,zorder=1)
      q=plt.quiver(lons_cen[::k],lats_cen[::k],rel_uc_flat[ks][::k,::k],rel_vc_flat[ks][::k,::k],pivot="mid",headwidth=3,width=0.01,scale=120,axes=ax,zorder=1)
    
    ax.quiverkey(q,0.8,-0.25,10,"10 m/s")
      
    fig.subplots_adjust(top=0.8,bottom=0.18,left=0.015,right=0.85,hspace=0.3,wspace=0.04)
    cax=fig.add_axes([0.88,0.11,0.05,0.73])
    fig.colorbar(a,cax=cax)
    
    fig.suptitle("Relevant flat regimes: " + case_label + "\nTarget date:  %04d-%02d-%02d - Fcst lead: %d"%(datet.year,datet.month,datet.day,lead),y=0.99,fontsize='x-large',fontweight='bold',x=0.45)
    plt.savefig(figure_path + "/Relevant_flat_regimes_" + fig_label + "_%04d-%02d-%02d_lead_%d.png"%(datet.year,datet.month,datet.day,lead))
    plt.show()
    
    
  #### tiered

    # build tiered regimes labels
    
    nregimes = {1:5,2:7,3:7,4:8,5:6,6:5,7:7,8:6}
    letters=['a','b','c','d','e','f','g','h']
    labels = [["%d%s"%(ki,letters[j]) for j in range(nregimes[ki])] for ki in range(1,9)]
    labels = [item for sublist in labels for item in sublist]
    

    fig=plt.figure()
    
    # determine best row/column structure
    
    nrows=1
    ncols=1
    
    if nr_flat>2:
        nrows=2
        ncols=round(nr_tiered/nrows)
        if ncols>4:
           nrows=3
           ncols=round(nr_tiered/nrows) 
    if nr_tiered>6:
        nrows=3
        ncols=round(nr_tiered/nrows)
        if ncols>4:
           nrows=4
           ncols=round(nr_tiered/nrows)
    if nr_tiered>9:
        nrows=4
        ncols=round(nr_tiered/nrows)
        if ncols>4:
           nrows=5
           ncols=round(nr_tiered/nrows)

    
    
    for ks in range(nr_tiered):

  # add every single subplot to the figure with a for loop

      # exceedance
      ax = fig.add_subplot(nrows,ncols,ks+1,projection=ccrs.PlateCarree())    
      plt.xlim(longitudes[0],longitudes[-1])
      plt.ylim(latitudes[-1],latitudes[0])     
      plt.title("Tier " + labels[fcst_tiered_regs[ks]-1] + " - prob [%]: "  + str(int(rel_prob_tiered[ks]*100)))
      a=plt.contourf(longitudes,latitudes,np.array(rel_exc_tiered[ks])*100,clevs,axes=ax,cmap=cmap, norm=norm)     
      ax.coastlines()
      # centroid winds
      k=12
      #q=plt.quiver(era5.extract('eastward_wind')[0][::k,::k],era5.extract('northward_wind')[0][::k,::k],pivot="mid",headwidth=3,width=0.01,scale=120,axes=ax,zorder=1)
      q=plt.quiver(lons_cen[::k],lats_cen[::k],rel_uc_tiered[ks][::k,::k],rel_vc_tiered[ks][::k,::k],pivot="mid",headwidth=3,width=0.01,scale=120,axes=ax,zorder=1)
    
    ax.quiverkey(q,0.8,-0.25,10,"10 m/s")
      
    fig.subplots_adjust(top=0.8,bottom=0.18,left=0.015,right=0.85,hspace=0.3,wspace=0.04)
    cax=fig.add_axes([0.88,0.11,0.05,0.73])
    fig.colorbar(a,cax=cax)
    
    fig.suptitle("Relevant tiered regimes: " + case_label + "\nTarget date:  %04d-%02d-%02d - Fcst lead: %d"%(datet.year,datet.month,datet.day,lead),y=0.99,fontsize='x-large',fontweight='bold',x=0.45)
    plt.savefig(figure_path + "/Relevant_tiered_regimes_" + fig_label + "_%04d-%02d-%02d_lead_%d.png"%(datet.year,datet.month,datet.day,lead))
    plt.show()
       
    return




######## ------------------------------------------------- END OF MODULES -------------------------------------------------------############

main()
