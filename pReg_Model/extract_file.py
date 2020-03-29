import os
import sys
import argparse
import numpy as np
from shutil import copyfile
gravity = [-625, -700, -750, -800]
gravity = list(range(-800,-599,25))
#time = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3,0.31, 0.32, 0.33, 0.35,0.37, 0.38,0.39, 0.4,0.42,0.43, 0.45,0.46, 0.5, 0.52,]
# time = np.arange(0.1, 0.7, 0.01)
# time = list(range(1, 60))
time = list(np.round(np.arange(0, 0.75, 0.01), 2))
# for i in range(len(time)):
#     time[i] = time[i] / 100
# print (time)
# atwood = list(np.round(np.arange(0.08, 0.5, 0.02), 2))
atwood = list(np.round(np.arange(0.02, 0.5, 0.02), 2))
amplitude = [0.1, 0.2, 0.3, 0.4, 0.5]


def main(out_dir, in_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dirr = [f for f in os.listdir(in_dir) if not os.path.isfile(os.path.join(in_dir, f))]

    for dirs in dirr:
        dirs = str(dirs)
        # print("len(gravity)")
        # print(len(gravity))
        for g in gravity:
            g = "gravity_" + str(g)
            # print("len(amplitude)")
            # print(len(amplitude))
            for amp in amplitude:
                amp = "amplitode_" + str(amp)
                # print("len(atwood)")
                # print(len(atwood))
                for at in atwood:
                    at = "atwood_" + str(at)
                    atwood_tmp = 'atwood_' + dirs.split('atwood_')[1]
                    
                    #print (atwood_tmp, at)
                    if dirs.__contains__(str(g)) and dirs.__contains__(str(amp)) and atwood_tmp == (str(at)):
                        # print("len(time)")
                        # print(len(time))
                        for t in time:
                            t1 = "time=" + str(t) + ".png"
                            dir1 = os.path.join(in_dir, dirs)
                            file_name = os.path.join(out_dir, dirs + "_time_" + str(t) + ".png")
                            src1 = os.path.abspath(os.path.join(os.path.abspath(dir1), t1))

                            if os.path.isfile(src1):
                                copyfile(src1, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs tests with varying input sizes.')
    parser.add_argument("-out_dir",
    
                        help='Path to the directory containing the runs.')
    parser.add_argument('-in_dir',
  
                        help='Path to the directory containing the runs.')
#    in_dir = "SimulAI/SimulationBW3"
    args = parser.parse_args()
    main(os.path.abspath(args.out_dir), os.path.abspath(args.in_dir))
