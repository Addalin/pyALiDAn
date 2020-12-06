"""
CHOOSE: telescope: far_range , METHOD: Klett_Method
each sample will have 60 bins (aka 30 mins length)
path to db:  stationdb_file


Date
start_time
end_time
period <30 min>
wavelength
method (KLET)

Y:
LC (linconst from .db)
LC std (un.linconst from.db)
r0 (from *_profile.nc
dr (r1-r0)

X:
att_bsc_path (station.lidar_src_folder/<%Y/%M/%d> +nc_zip_file+'att_bsc.nc'
mol_path ( station .molecular_src_folder/<%Y/%M><day_mol.nc>
"""

