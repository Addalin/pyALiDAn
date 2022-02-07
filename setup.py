from setuptools import setup
setup(
    name='learning_lidar',
    version='0.1',
    packages=['data', 'learning_lidar', 'learning_lidar.utils', 'learning_lidar.dataseting',
              'learning_lidar.generation', 'learning_lidar.preprocessing', 'learning_lidar.learning_phase',
              'learning_lidar.learning_phase.models', 'learning_lidar.learning_phase.utils_',
              'learning_lidar.learning_phase.data_modules'],
    url='https://github.com/Addalin/learning_lidar',
    license='MIT',
    author='Adi Vainiger',
    author_email='adi.vainiger@gmail.com , addalin@campus.technion.ac.il',
    description='Learning Lidar is an atmospheric lidar learning repo. The repo includes:'
                '1. ALiDAn: Atmospheric Lidar Data Augmentation. Statistical data generation'
                '2. Preprocessing of lidar data'
                '2. Learning pipeline for lidar analysis'
                'This project uses data from several databases:'
                '1. Lidar database produced by the PollyNet Processing chain by TROPOS:'
                '    https://github.com/PollyNET/Pollynet_Processing_Chain '
                ' 2. AERONET by NASA: https://aeronet.gsfc.nasa.gov/new_web/data.html'
                ' 3. GDAS by NOAA: https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation'
                ' * Additional Databases are possible as well. More details can be found in the ALiDAn manuscript'
)
# TODO: update information on ALiDAn