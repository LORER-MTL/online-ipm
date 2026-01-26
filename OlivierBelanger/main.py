import argparse
from oop_opt.utils.utils import *

def main(W_comp, reuse_data=False, data_file='simulation_data.pkl'):
    if W_comp:
        run_w_comp()
    else:
        run_scenario_comp(reuse_data, data_file)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--W_comp', action='store_true', help='Run W comparison')
    parser.add_argument('--reuse_data', action='store_true', help='Reuse old data for plotting')
    parser.add_argument('--data_file', type=str, default='sim_data_20240819_131147.dat', help='File to save/load simulation data')
    args = parser.parse_args()
    
    main(args.W_comp, args.reuse_data, args.data_file)