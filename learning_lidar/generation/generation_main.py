import generate_density
from learning_lidar.utils.utils import get_base_arguments

parser = get_base_arguments()

parser.add_argument('--plot_results', action='store_true',
                    help='Whether to plot graphs')

parser.add_argument('--save_ds', action='store_true',
                    help='Whether to save the datasets')

args = parser.parse_args()


generate_density.generate_density_main(args)
