
## This is a short explanation of how to generate background lidar signal
1. Upload a background noise image as [this]('C:\Users\addalin\Dropbox\Lidar\code\background_signal\background_noise.jpg') one from 04-Apr-2017 to [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).
2. Generate dataset samples for each curve. For the above example I created 2 datasets for each channel (R,G,B). One of them samples the higher bound of the data, and the other samples the lower curve. 
3. Remember to give meaningful names to the datasets.  
4. Then it is possible to download the data as csv format, and also open it in [chart-studio](https://chart-studio.plotly.com/create/#/) of [plotly](https://plotly.com/).
5. When opening in chart-studio, there is an easy way to estimate a curve fit to each one of the datasets. Gaussian curve looks more appropriate.
   These are the estimated parameters:
    - A: Bias term (above y=0)
	- H: The height of Gaussian curve
	- t0: The center of the Gaussian lobe
	- W: The width of Gaussian lobe
	- the curve model follows: $y = A+H*np.exp(-(t-t0)**2/(2*(W**2))) \end {equation}$\\
	- Note: the 
7. The parameters are saved in [curve_params.yml](C:\Users\addalin\Dropbox\Lidar\code\background_signal\curve_params.yml).
    The file contains the following fields:
    ' UV:
        high:
            A: 0.03
            H: 1.12
            t0: 342.5
            W: 115 
        low:
            A: 0.0001
            H: 0.87
            t0: 342.5
            W: 100
        C: 'darkblue' # the color of the samples 
        C_new: 'blueviolet''' # the color of the bounding curves
    '
8. The initial samples of the dataset saved in [bg_signal_measurments.csv](C:\Users\addalin\Dropbox\Lidar\code\background_signal\bg_signal_measurments.csv)  datasets common csv file with the following fields: 
UV_high_times, UV_high_vals,	UV_low_times, UV_low_vals,	G_high_times,	G_high_vals,	G_low_times,	G_low_vals,	IR_high_times,	IR_high_vals,	IR_low_times,	IR_low_vals
9. The values from the csv files are presented in the image as dots, and the values from the yaml files are to create the bounding gauss curve of each channel.
