import calendar
import os
from datetime import datetime, timedelta

import astral
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml
from astral.sun import sun
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from learning_lidar.generation import generation_utils as gen_utils, generate_bg_signals_utils as gen_bg_utils
from learning_lidar.utils import misc_lidar, xr_utils, vis_utils, global_settings as gs

sns.set_theme()
vis_utils.set_visualization_settings()
sns.set_palette(sns.color_palette("tab10"))


class BackgroundGenerator:
    def __init__(self, station_name):
        self.station = station = gs.Station(station_name=station_name)
        self.wavelengths = gs.LAMBDA_nm().get_elastic()
        self.eps = np.finfo(np.float).eps

        data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))), 'data')
        self.bg_signal_folder = os.path.join(data_folder, 'background_signal')

        # #### 1. Load parameters of gaussian curve fit
        bg_params_path = os.path.join(self.bg_signal_folder, 'curve_params.yml')
        with open(bg_params_path, 'r') as ymlfile:
            self.params = yaml.safe_load(ymlfile)

        # #### 2.Time resolution adjustments of the curve fitting
        # - The ratio is to adjust time bins $t$ to lidar sampling bins (delta_t=30[s] )
        # - the estimated curve refer to delta_t=100[s], self.bins_per_day=864
        self.bins_per_day = station.total_time_bins
        self.t = np.arange(0, self.bins_per_day)
        self.bins_ratio = self.bins_per_day / self.params['bins_per_day']
        self.factors = [1 / 2, 1, 1 / 7]  # This are the original factor of the data for [UV,G,IR]

    def generate_bg(self, start_day, end_day, high_curves, low_curves, mean_curves, ds_year,
                    day_bins_length_0, tbin_noon_tst_0, sun_noons):
        # %%Calculating Gaussian curve parameters for any day of 2017
        # #TODO: this section requires massive re-organisation and commenting

        max_new_curves = []
        min_new_curves = []
        mean_new_curves = []
        mean_new_signal = []
        for i, (curve_h, curve_l, mean_val, channel) in enumerate(
                zip(high_curves, low_curves, mean_curves, ['UV', 'G', 'IR'])):
            params_h = self.params[channel]['high']
            params_l = self.params[channel]['low']
            t0 = self.bins_ratio * params_l['t0'] - day_bins_length_0 / 2
            t1 = self.bins_ratio * params_l['t0'] + day_bins_length_0 / 2

            def calc_curves(prms, curve):
                A_ds = xr.apply_ufunc(lambda x: prms['A'] * self.factors[i] * x, ds_year.relevation, keep_attrs=True)
                H_ds = xr.apply_ufunc(lambda x: prms['H'] * self.factors[i] * x, ds_year.relevation, keep_attrs=True)

                # calculating dx for specific day

                dx = (tbin_noon_tst_0 - (self.bins_ratio * prms['t0'])) / day_bins_length_0

                t0_bound = [(gen_bg_utils.dt2binscale(tnoon) - dx * daylight)
                            for tnoon, daylight in zip(sun_noons, ds_year.daylightbins.values)]

                t0_ds = H_ds.copy(deep=True)
                t0_ds.data = t0_bound

                photons_twilight = 0.5 * (curve[int(t1)] + curve[int(t0)])

                max_val_ds = xr.apply_ufunc(lambda ratio: (prms['H'] + prms['A']) * ratio * self.factors[i],
                                            ds_year.relevation, keep_attrs=True)

                min_val_ds = xr.apply_ufunc(lambda ratio: (prms['A']) * ratio * self.factors[i],
                                            ds_year.relevation, keep_attrs=True)

                W_ds = xr.apply_ufunc(lambda min_val, max_val, daylight:
                                      gen_bg_utils.calc_gauss_width(min_val, max_val, photons_twilight, daylight),
                                      min_val_ds, max_val_ds, ds_year.daylightbins, keep_attrs=True)

                bound = np.array([misc_lidar.calc_gauss_curve(self.t, A, H, t0_, W)
                                  for A, H, t0_, W in zip(A_ds.values, H_ds.values, t0_ds.values, W_ds.values)])

                return bound

            max_new = calc_curves(prms=params_h, curve=curve_h)  # High curves
            min_new = calc_curves(prms=params_l, curve=curve_l)  # low curves
            max_new_curves.append(max_new)
            min_new_curves.append(min_new)

            """Calculating mean curves & generating new averaged bg signal """
            mean_new = (0.5 * (max_new + min_new))
            mean_new_curves.append(mean_new)
            std_val = (0.5 * (max_new - min_new))
            bg_new_sig = np.zeros((len(ds_year.Time), self.bins_per_day))
            for day in range(len(ds_year.Time)):
                bg_new_sig[day, :] = (mean_new[day].reshape(self.bins_per_day, 1) + np.diagflat(std_val[day])
                                      @ np.random.randn(self.bins_per_day, 1)).T
            bg_new_sig[bg_new_sig < 0] = self.eps
            mean_new_signal.append(bg_new_sig)
        # %% Generating background dataset per year
        ds_bins = pd.date_range(start=start_day,
                                end=end_day,
                                freq='30S',
                                tz='UTC')
        bg_channels = []
        for ind, wavelength in enumerate(self.wavelengths):
            bg_channels.append(xr.Dataset(data_vars={'bg': ('Time', mean_new_signal[ind].flatten())},
                                          coords={'Time': ds_bins.values, 'Wavelength': wavelength}))
        ds_bg_year = xr.concat(bg_channels, dim='Wavelength')
        ds_bg_year.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$\rm nm$'}
        ds_bg_year.bg.attrs = {'long_name': r'$<p_{\rm bg}>$', 'units': r'$\rm photons$',
                               'info': 'Daily averaged background signal'}

        return ds_bg_year

    def generate_curves(self, day_0):
        bg_new = []
        high_curves = []
        low_curves = []
        mean_curves = []
        day_times_day0 = pd.date_range(start=day_0, end=(day_0 + timedelta(hours=24) - timedelta(seconds=30)),
                                       freq='30S',
                                       tz='UTC')
        df_day0 = pd.DataFrame(data=day_times_day0, columns=['Time'])
        for chan, factor in zip(['UV', 'G', 'IR'], self.factors):
            prms_h = self.params[chan]['high']
            prms_l = self.params[chan]['low']
            ''' Calculating bounding curves series'''
            high_curve = factor * misc_lidar.calc_gauss_curve(self.t, prms_h['A'], prms_h['H'],
                                                              self.bins_ratio * prms_h['t0'],
                                                              self.bins_ratio * prms_h['W'])
            low_curve = factor * misc_lidar.calc_gauss_curve(self.t, prms_l['A'], prms_l['H'],
                                                             self.bins_ratio * prms_l['t0'],
                                                             self.bins_ratio * prms_l['W'])
            high_curves.append(high_curve)
            low_curves.append(low_curve)
            ''' Generating new random series of background signal'''
            mean_val = 0.5 * (low_curve + high_curve)
            mean_curves.append(mean_val)
            std_val = 0.5 * (high_curve - low_curve)
            bg_new_chan = mean_val.reshape(self.bins_per_day, 1) + np.matmul(np.diagflat(std_val),
                                                                             np.random.randn(self.bins_per_day, 1))
            bg_new.append(bg_new_chan)
        bg_new = np.array(bg_new)
        bg_new[bg_new < 0] = self.eps
        bgmean = xr.DataArray(dims=('Wavelength', 'Time'), data=bg_new.reshape((3, self.bins_per_day)))
        bgmean = bgmean.assign_coords({"Wavelength": self.wavelengths, "Time": df_day0.Time.values})
        bgmean.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$\rm nm$'}
        bgmean.attrs = {'info': 'Background Mean Signal',
                        'long_name': r'$<p_{bg}>$',
                        'units': r'$photons$',
                        'name': 'pbg'}

        bg_mean = bgmean.copy(deep=True).assign_attrs({'info': 'Mean Background Signal'})
        bg_mean.data = np.array(mean_curves).reshape((3, self.bins_per_day))
        # TODO bg_mean vs bgmean - what is it supposed to be?

        return bgmean, high_curves, low_curves, mean_curves

    def bg_signals_generation_main(self, plot_results=True):
        # ## Daily Sun Elevation
        # - Explore daily and yearly $\alpha_{\rm sun}$

        cur_day = datetime(2017, 10, 4)
        day_0 = datetime(2017, 4, 4)
        timezone = 'UTC'
        cur_day_str = cur_day.strftime('%Y-%m-%d')
        loc = astral.LocationInfo("Haifa", 'Israel', timezone, self.station.lat, self.station.lon)
        loc.observer.elevation = self.station.altitude

        # #### 1. Sun elevation during the year :
        # - Extracting the sun elevation at noon time for every day during 2017

        # Sun elevation at noon time per day:
        year_days = pd.date_range(start=datetime(2017, 1, 1), end=datetime(2017, 12, 31), freq='D', tz='UTC')
        sun_noons = [sun(loc.observer, c_day)['noon'] for c_day in year_days]
        sun_elevations = [astral.sun.elevation(loc.observer, dt) for dt in sun_noons]
        ds_year = xr.Dataset(data_vars={'sunelevation': ('Time', sun_elevations)},
                             coords={'Time': year_days.values})
        ds_year.sunelevation.attrs = {'long_name': r'$\alpha_{sun}$', 'units': r'$^{\circ}$'}
        ds_year.Time.attrs = {"units": fr"{timezone}"}

        if plot_results:
            gen_bg_utils.plot_sun_elevation_at_noon_times(ds_year)

        # #### 2. Sun elevation during the day :

        # Create daily dataset of sun elevation, azimuth, daylight hours
        day_sun = sun(loc.observer, cur_day)
        day_times = pd.date_range(start=cur_day,
                                  end=(cur_day + timedelta(hours=24) - timedelta(seconds=30)),
                                  freq='30S', tz='UTC')
        df = pd.DataFrame(data=day_times, columns=['Time'])
        ds_day = xr.Dataset(data_vars=
                            {'sunazimuth': ('Time', df.Time.apply(lambda x: astral.sun.azimuth(loc.observer, x))),
                             'sunelevation': ('Time', df.Time.apply(lambda x: astral.sun.elevation(loc.observer, x)))},
                            coords={'Time': df.Time.values})
        ds_day.sunelevation.attrs = {'long_name': r'$\alpha_{sun}$', 'units': r'$^{\circ}$'}
        ds_day.Time.attrs = {"units": fr"{timezone}"}

        if plot_results:
            gen_bg_utils.fit_curve_and_plot_sun_elevation_during_day(loc, day_sun, day_0, ds_day, self.bins_per_day)

        # #### 3. Solar Irradiance vs. Solar Elevation
        #  - Fitting irradiance according to :$f(\alpha_{sun}) = a\cdot \cos(b \cdot \alpha_{sun} + c) + d$

        # Load irradiance - solar elevation data

        csv_iradiance_elevation = r'irradiance_solarElevation.csv'
        csv_name = os.path.join(self.bg_signal_folder, csv_iradiance_elevation)
        df_iradiance_solar = pd.read_csv(csv_name)
        df_iradiance_solar.rename(columns={'X_vector_irradiance': r'$\alpha_{\rm sun}$',
                                           'Y_vector_irradiance': 'vector-irradiance'},
                                  inplace=True)
        X = df_iradiance_solar[r'$\alpha_{\rm sun}$'].values.reshape(-1, 1)
        Y = df_iradiance_solar['vector-irradiance'].values.reshape(-1, 1)
        # %% Linear fit :
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X[:], Y[:])  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        # %% Cos() - fit :
        popt, pcov = curve_fit(gen_bg_utils.func_cos, X[:, 0], Y[:, 0])

        if plot_results:
            angles = np.linspace(0, 90, 100, endpoint=True)
            cos_fit = gen_bg_utils.func_cos(angles, popt[0], popt[1], popt[2], popt[3])
            # %% Plot data + fitting curves
            fig, ax = plt.subplots(ncols=1, nrows=1)
            ax.plot(X, Y_pred, color='magenta', label='linear-fit', linestyle='--')
            ax.plot(angles, cos_fit, color='c', label='cos-fit')
            df_iradiance_solar.plot(x=r'$\alpha_{\rm sun}$',
                                    y='vector-irradiance',
                                    style='o',
                                    c="darkblue",
                                    ax=ax)
            plt.ylabel(r'Irradiance')
            plt.legend()
            plt.xlim([-1, 91])
            plt.ylim([-0.02, 1.1])
            plt.tight_layout()
            plt.show()

        # ## Daily background
        # - Create a mean background noise $<p_{bg}>$

        # #### 3. Calculate bg mean of the original day - 04/04/2014

        # %%Generate background signal'''
        bgmean, high_curves, low_curves, mean_curves = self.generate_curves(day_0)
        if plot_results:
            gen_bg_utils.plot_daily_bg_signal(bgmean, high_curves, low_curves, mean_curves, self.bins_per_day)

        # #### 4. Calculating Gaussian curve parameters for a new day:
        # - The mean is set based on the time shifting of the mean curves from 04/04/2017:
        # $\Delta t_{\rm shift} = \frac{T_{\rm noon0} - t_{0-curve}}{\Delta t_{\rm daylight0}}$
        # $T_{\rm new-curve} = T_{\rm noon} - \Delta t_{\rm daylight} \cdot \Delta t_{\rm shift}$
        # - The amplitude is set according to the relation of angle-irradiance

        # Shifted mean of the curve
        tbin_dusk = gen_bg_utils.dt2binscale(day_sun['dusk'])
        tbin_dawn = gen_bg_utils.dt2binscale(day_sun['dawn'])
        tbin_noon_tst = gen_bg_utils.dt2binscale(day_sun['noon'])
        day_bins_length = (tbin_dusk - tbin_dawn)

        day_sun0 = sun(loc.observer, day_0)
        day_bins_length_0 = gen_bg_utils.dt2binscale(day_sun0['dusk']) - gen_bg_utils.dt2binscale(day_sun0['dawn'])
        tbin_noon_tst_0 = gen_bg_utils.dt2binscale(day_sun0['noon'])

        # Calc elevation ratio
        max_elevation_0 = astral.sun.elevation(loc.observer, day_sun0['noon'])

        day_max_elevation = ds_day.sunelevation.values.max()
        iradiance_orig_cos = gen_bg_utils.func_cos(max_elevation_0, popt[0], popt[1], popt[2], popt[3])
        iradiance_day_cos = gen_bg_utils.func_cos(day_max_elevation, popt[0], popt[1], popt[2], popt[3])

        ratio_elevation = iradiance_day_cos / iradiance_orig_cos
        print(f"Ration elevation for {cur_day_str}: {ratio_elevation:.2f}")

        # %% Calculate curves parameters
        max_new_curves = []
        min_new_curves = []
        mean_new_curves = []
        mean_new_signal = []
        for i, (curve_h, curve_l, mean_val, c, chan) in enumerate(
                zip(high_curves, low_curves, mean_curves, vis_utils.COLORS, ['UV', 'G', 'IR'])):
            prms_h = self.params[chan]['high']
            prms_l = self.params[chan]['low']
            t0 = self.bins_ratio * prms_l['t0'] - day_bins_length_0 / 2
            t1 = self.bins_ratio * prms_l['t0'] + day_bins_length_0 / 2

            A_l = prms_l['A'] * ratio_elevation * self.factors[i]
            H_l = prms_l['H'] * ratio_elevation * self.factors[i]

            # calculating dx_l for 4/4/2017
            dx_l = (tbin_noon_tst_0 - (self.bins_ratio * prms_l['t0'])) / day_bins_length_0
            t0_l = tbin_noon_tst - dx_l * day_bins_length

            photons_twilight_l = 0.5 * (curve_l[int(t1)] + curve_l[int(t0)])
            max_val = (prms_l['H'] + prms_l['A']) * ratio_elevation * self.factors[i]
            min_val = (prms_l['A']) * ratio_elevation * self.factors[i]
            W_l = gen_bg_utils.calc_gauss_width(min_val, max_val, photons_twilight_l, day_bins_length)

            min_new = misc_lidar.calc_gauss_curve(self.t, A_l, H_l, t0_l, W_l)
            min_new_curves.append(min_new)
            print(f"{chan} low: A={A_l:.2f}, H={H_l:.2f}, W={W_l:.2f},"
                  f" t0={t0_l:.2f},W_dt={gen_bg_utils.binscale2dt(W_l, cur_day).strftime('%H:%M:%S')} ,"
                  f"t0_dt:{gen_bg_utils.binscale2dt(t0_l, cur_day).strftime('%H:%M:%S')}")

            A_h = prms_h['A'] * ratio_elevation * self.factors[i]
            H_h = prms_h['H'] * ratio_elevation * self.factors[i]

            # calculating dx_h for 4/4/2017
            dx_h = (tbin_noon_tst_0 - (self.bins_ratio * prms_h['t0'])) / day_bins_length_0
            t0_h = tbin_noon_tst - dx_h * day_bins_length

            photons_twilight_h = 0.5 * (curve_h[int(t1)] + curve_h[int(t0)])
            max_val = (prms_h['H'] + prms_h['A']) * ratio_elevation * self.factors[i]
            min_val = (prms_h['A']) * ratio_elevation * self.factors[i]
            W_h = gen_bg_utils.calc_gauss_width(min_val, max_val, photons_twilight_h, day_bins_length)

            max_new = misc_lidar.calc_gauss_curve(self.t, A_h, H_h, t0_h, W_h)
            max_new_curves.append(max_new)
            print(f"{chan} high: A={A_h:.2f}, H={H_h:.2f}, W={W_h:.2f},"
                  f" t0={t0_h:.2f}, W_dt={gen_bg_utils.binscale2dt(W_h, cur_day).strftime('%H:%M:%S')} ,"
                  f" t0_dt:{gen_bg_utils.binscale2dt(t0_h, cur_day).strftime('%H:%M:%S')}")

            mean_new = 0.5 * (max_new + min_new)
            mean_new_curves.append(mean_new)

            std_val = 0.5 * (max_new - min_new)
            bg_new_sig = mean_new.reshape(self.bins_per_day, 1) + np.matmul(np.diagflat(std_val),
                                                                            np.random.randn(self.bins_per_day, 1))
            mean_new_signal.append(bg_new_sig)

            del min_new, max_new, mean_new, bg_new_sig
        # %% Generate dataset per type of curve
        bg_max_new = bgmean.copy(deep=True)
        bg_max_new.attrs['info'] = 'Background Max Signal'
        bg_max_new.data = np.array(max_new_curves).reshape((3, self.bins_per_day))
        bg_min_new = bgmean.copy(deep=True)
        bg_min_new.attrs['info'] = 'Background Min Signal'
        bg_min_new.data = np.array(min_new_curves).reshape((3, self.bins_per_day))
        bg_mean_new = bgmean.copy(deep=True)
        bg_mean_new.attrs['info'] = 'Background Mean Signal'
        bg_mean_new.data = np.array(mean_new_curves).reshape((3, self.bins_per_day))
        bg_signal_new = bgmean.copy(deep=True)
        bg_signal_new.attrs['info'] = 'Background Mean Signal'
        bg_signal_new.data = np.array(mean_new_signal).reshape((3, self.bins_per_day))
        # %%Plot the new curves
        if plot_results:
            fig, ax = plt.subplots(ncols=1, nrows=1)
            for i, (curve_h, curve_l, mean_val, c, chan, wavelength) in enumerate(
                    zip(high_curves, low_curves, mean_curves, vis_utils.COLORS, ['UV', 'G', 'IR'], self.wavelengths)):
                ax.fill_between(bgmean.Time, curve_h, curve_l, alpha=.2, color=c)

            bg_min_new.plot(hue='Wavelength', ax=ax, linewidth=0.8, linestyle='--')
            bg_max_new.plot(hue='Wavelength', ax=ax, linewidth=0.8, linestyle='--')
            SHOW_MEAN = False
            if SHOW_MEAN:
                bg_mean_new.plot(hue='Wavelength', ax=ax, linewidth=0.8)
            SHOW_NEW = False
            if SHOW_NEW:
                bg_signal_new.plot(hue='Wavelength', ax=ax, linewidth=0.8)
            ax.xaxis.set_major_formatter(vis_utils.TIMEFORMAT)
            ax.xaxis.set_tick_params(rotation=0)
            ax.set_ybound([-.01, 2])
            ax.set_xlim([bg_max_new.Time.values[0], bg_max_new.Time.values[-1]])
            plt.tight_layout()
            plt.show()

        # #### 4. Calculating Gaussian curve parameters for any day of 2017

        # ### 1. Irradiance vs sun elevation at noon times

        # %% Irradiance vs sun elevation
        ds_year = ds_year.assign(irradiance=xr.apply_ufunc(lambda alpha:
                                                           gen_bg_utils.func_cos(alpha, popt[0], popt[1], popt[2],
                                                                                 popt[3]),
                                                           ds_year.sunelevation, keep_attrs=True))

        ds_year.irradiance.name = 'irradiance'
        ds_year.irradiance.attrs = {'long_name': 'irradiance'}

        if plot_results:
            gen_bg_utils.plot_irradiance_vs_sun_elevation_at_noon_times(ds_year)

        # ### 2. Ratio elevation per day (at noon time)
        # Ratio elevation per day
        irradiance_0 = gen_bg_utils.func_cos(max_elevation_0, popt[0], popt[1], popt[2], popt[3])
        ds_year = ds_year.assign(relevation=xr.apply_ufunc(lambda x, y: x / y, ds_year.irradiance,
                                                           irradiance_0, keep_attrs=True))
        ds_year.relevation.name = 'ratio elevation'
        ds_year.relevation.attrs = {'long_name': 'ratio elevation'}
        if plot_results:
            ds_year.relevation.plot()
            plt.show()

        # ### 3. Daylight time (hours) per day
        # > Or bins where there are 120 bins per one hour.

        # Daylight time per day
        def sun_dt(loc, day, type_):
            return sun(loc.observer, day)[type_]

        daylight_timedelta = np.array([sun_dt(loc, c_day, 'dusk') - sun_dt(loc, c_day, 'dawn') for c_day in year_days])
        daylight_time = np.array([gen_bg_utils.dt_delta2time(dt) for dt in daylight_timedelta])
        daylight_bins = np.array([gen_bg_utils.dt2binscale(dt) for dt in daylight_time])
        ds_year = ds_year.assign(daylightbins=xr.Variable(dims={'Time': ds_year.Time},
                                                          data=daylight_bins,
                                                          attrs={'long_name': 'Daylight bins',
                                                                 'info': 'dDaylight bins count per day'}),
                                 daylighthrs=xr.Variable(dims={'Time': ds_year.Time}, data=daylight_bins / 120,
                                                         attrs={'long_name': 'Daylight hours',
                                                                'info': 'Daylight hours per day'}))
        if plot_results:
            ds_year.daylighthrs.plot()
            plt.show()
        # %%

        # ### 4. Calculating Gaussian curve parameters for any day of 2017
        """
        <MAIN>
        # TODO:
        
        A ) create a function that get:
        
            1.per wavelength:[(pbg-night/max) A +-da,
                              (pbg-noon/min) H+-dh
                              relative different of pick time from noon time,
                              pbg-TW +- dpgbtw ]
                              
            2. tdawn, tdust,tnoon, sun angle thruog time of the day,
        
        B ) helper function: calc_gauss_width, calc_gauss_curve
        
        C ) generate the above parameters based on a single example. 
        """

        start_day = datetime(2017, 1, 1)
        end_day = datetime(2018, 1, 1) - timedelta(seconds=self.station.freq)
        ds_bg_year = self.generate_bg(start_day, end_day, high_curves, low_curves, mean_curves, ds_year,
                                      day_bins_length_0, tbin_noon_tst_0, sun_noons)

        if plot_results:
            # Plot current day
            gen_bg_utils.plot_bg_one_day(ds_bg_year, c_day=cur_day)

            c_day = datetime(2017, 1, 1)
            # Plot 1st half of the year
            gen_bg_utils.plot_bg_part_of_year(ds_bg_year,
                                              dslice=slice(c_day, c_day + timedelta(days=181) - timedelta(seconds=30)))

            # Plot 2nd half of the year
            gen_bg_utils.plot_bg_part_of_year(ds_bg_year,
                                              dslice=slice(c_day, c_day + timedelta(days=184) - timedelta(seconds=30)))

        # save yearly dataset
        gen_utils.save_full_params_dataset(self.station, start_day, end_day, ds_bg_year, type_='bg')

        # %% Save monthly bg dataset
        for month in range(1, 12 + 1):
            gen_utils.save_monthly_params_dataset(self.station, start_day.year, month, ds_bg_year, type_='bg')

        #  TODO
        # #### 1. Adjust section 4 - to run for any given period with start_day, end_day
        # #### 2. Save the curve parameters per day : A_l, H_l, W_l, t0_l and  A_h, H_h, W_h, t0_h
        # > A_l_ds, H_l_ds, W_l_ds , t0_l_ds, A_h_ds, H_h_ds, W_h_ds , t0_h_ds
        # #### 2.  Save the bounding curves per day :  max_new_curves, min_new_curves, mean_new_curves
        # > similar to the way ds_bg_year was generated


if __name__ == '__main__':
    bg_generator = BackgroundGenerator(station_name='haifa_shubi')
    bg_generator.bg_signals_generation_main()
