import sqlite3
import pandas as pd


def query_database(query="SELECT * FROM lidar_calibration_constant;", database_path="pollyxt_tropos_calibration.db"):
    """
    Query is a string following sqlite syntax (https://www.sqlitetutorial.net/) to query the .db
    Examples:
    query_basic = "
    SELECT * -- This is a comment. Get all columns from table
    FROM lidar_calibration_constant -- Which table to query
    ;
    "

    query_advanced = "
    SELECT lcc.id, lcc.liconst, lcc.cali_start_time, lcc.cali_stop_time -- get only some columns
    FROM lidar_calibration_constant as lcc
    WHERE -- different filtering options on rows
        wavelength == 1064 AND
        cali_method LIKE 'Klet%' AND
        (cali_start_time BETWEEN '2017-09-01' AND '2017-09-02');
    "
    """
    # Connect to the db and query it directly into pandas df.
    with sqlite3.connect(database_path) as c:
        # Query to df
        # optionally parse 'id' as index column and 'cali_start_time', 'cali_stop_time' as dates
        df = pd.read_sql(sql=query, con=c, index_col='id', parse_dates=['cali_start_time', 'cali_stop_time'])

    return df


if __name__ == "__main__":
    wavelength = 1064

    query_advanced = f"""
    SELECT lcc.id, lcc.liconst, lcc.cali_start_time, lcc.cali_stop_time
    FROM lidar_calibration_constant as lcc
    WHERE
        wavelength == {wavelength} AND
        cali_method LIKE 'Klet%' AND
        (cali_start_time BETWEEN '2017-09-01' AND '2017-09-02');
    """
    df = query_database(query=query_advanced)
    print(df)
