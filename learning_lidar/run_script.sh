#cd generation  || exit
#python daily_signals_generation.py --start_date 2017-04-01 --end_date 2017-05-31 --save_ds --update_overlap_only
#python daily_signals_generation.py --start_date 2017-09-01 --end_date 2017-10-31 --save_ds --update_overlap_only

cd dataseting || exit
python dataseting.py --start_date 2017-04-01 --end_date 2017-05-31 --use_km_unit --calc_generated_stats
python dataseting.py --start_date 2017-04-01 --end_date 2017-10-31 --use_km_unit --calc_generated_stats

