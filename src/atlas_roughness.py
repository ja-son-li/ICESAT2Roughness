import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
import icepyx as ipx
import os
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
from geopy.distance import geodesic

class ATL06PassRoughness:
    """
    Read ATL06Pass from ICESat-2

    ARGUMENTS:
    -----------
    file_name (str) : file path of ATL06 file 
    roughness_length_scales (1, n_length_scales) : array of length scales (km) to perform roughness calculation
    """

    def __init__(self, file_name, roughness_length_scales=[25.]):
        self.roughness_length_scales = roughness_length_scales
        self.file_name = file_name

        reader = ipx.Read(data_source=file_name)
        reader.vars.append(beam_list=['gt1l'], var_list=['h_li',"sigma_geo_h", "latitude", "longitude"])
        ds = reader.load()

        # save off array 
        self.lats = ds['latitude'].values.flatten()
        self.lons = ds['longitude'].values.flatten()
        self.heights = ds['h_li'].values.flatten()
        self.height_uncertainties = ds['sigma_geo_h'].values.flatten()
        self.times = ds['delta_time'].values.flatten()

        #calculate distance along track
        self.distance_along_track = self.calculate_distance_along_track(self.lats, self.lons)

        # calculate roughness 
        self.roughness_values = self.calculate_roughness(roughness_length_scales)

    def calculate_roughness(self,roughness_length_scales):
        """
        ARGUMENTS:
        -----------
        roughness_length_scales (1, n_length_scales) : array of length scales (km) to perform roughness calculation

        OUTPUS:
        ----------
        roughness_values (n_length_scales, n_obs) : roughness value at observation point and length scale (m)
        """
        roughness_length_scales = np.arange(0.5,25, 0.5)
        roughness_values = np.full((len(roughness_length_scales), len(self.distance_along_track)), np.nan)

        for ind, track_dist in enumerate(self.distance_along_track):
            center_distance = track_dist

            for length_ind, roughness_length in enumerate(roughness_length_scales):

                # get the rolling window of retrieved photon heights 
                mask = (self.distance_along_track <= (center_distance+ roughness_length/2)) & (self.distance_along_track >= (center_distance - roughness_length/2)) 
                track_distances_in_window = self.distance_along_track[mask]

                # if the found endpoints of this window are not within 0.1 km of the desired endpoints, this window is not valid   
                if ( np.abs(np.max(track_distances_in_window) - (center_distance+ roughness_length/2))  > 0.1) | \
                    ( np.abs(np.min(track_distances_in_window) - (center_distance-roughness_length/2)) > 0.1):
                    break
                
                # take the standard deviation 
                roughness_values[length_ind,ind] = np.std(self.heights[mask])
            return roughness_values
        
    def save_to_xarray(self, save_directory:str):
        """
        Save the roughness values to a netcdf 

        Inputs:
        """

        #  create xarray dataset 
        ds = xr.Dataset(data_vars=dict(
            photon_height=(["photon_idx"], self.heights, {'units':'m'}),
            photon_height_uncertainty =(["photon_idx"], self.height_uncertainties, {'units':'m'}),
            surface_roughness = (["length_scale", "photon_idx"], self.roughness_values, {'units':'m'})),
            
            coords=dict(
                time = (["photon_idx"], self.times, {'time zone':'UTC'}),
                latitude = (["photon_idx"], self.lats, {'units':'WGS-84 degree'}),
                longitude = (["photon_idx"], self.lons, {'units':'WGS-84 degree'}),
                length_scales = (["length_scale"], self.roughness_length_scales, {'units':'km'})
            )
            )
        # save as netcdf 
        new_file_name = os.path.splitext(os.path.basename(self.file_name))[0] + '_roughness.nc'
        save_path = os.path.join(save_directory, new_file_name)
        ds.to_netcdf(save_path, mode = 'w', engine = 'netcdf4')

        return 
    
    def get_gridded_roughness(self):
        """
        
        """        
        distance_along_track_gridded = np.arange(0,np.max(self.distance_along_track),1) # km steps along track
        lon_grid = np.full( len(distance_along_track_gridded),np.nan)
        lat_grid =  np.full( len(distance_along_track_gridded),np.nan)
        roughness_values_grid = np.full((len(self.roughness_length_scales), len(distance_along_track_gridded)), np.nan)

        for ind, track_dist in enumerate(distance_along_track_gridded):
            # is there any data point at this distance along the track
            nearest_track_ind = np.argmin(np.abs(track_dist - self.distance_along_track))
            if np.abs(self.distance_along_track[nearest_track_ind]-track_dist) < 0.1:
                lat_grid[ind] = self.lats[nearest_track_ind]
                lon_grid[ind] = self.lons[nearest_track_ind]
                roughness_values_grid[:,ind] = self.roughness_values[:,nearest_track_ind]
        
        return lon_grid, lat_grid, roughness_values_grid
    
    @staticmethod
    def calculate_distance_along_track(lats, lons):
        """
        Calculate geodesic distance from first lat/lon point in array


        """
        lat_lon_zip = zip(lats, lons)
        start_lat_lon = (lats[0],lons[0])

        distance_along_track = np.zeros(np.size(lats))
        for ind,lat_lon in enumerate(lat_lon_zip):
            distance_along_track[ind] = geodesic(start_lat_lon, lat_lon).km

        return distance_along_track
