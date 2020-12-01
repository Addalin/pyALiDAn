from datetime import datetime
import global_settings as gs
import preprocessing as prep
start_day =  datetime(2019,4,3)
end_day = datetime(2019,4,20)
haifa_station = gs.Station(station_name='haifa_shubi')

chunk_paths = prep.convert_periodic_gdas(haifa_station,start_day, end_day)
