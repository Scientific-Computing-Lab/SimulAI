import shutil
import numpy as np
import argparse
import os


def get_time(path):
    t1 = "".join(path.split('.png'))

    t1 = t1.split("time")[1]
    t1 = "".join(t1[1:])
    t2 = t1.split(".")
    if len(t2[1]) == 1:
        t2[1] = t2[1] + "0"
    t1 = t2[0] + t2[1]
    return int(t1)


def main(in_dir, out_dir):
    t_fac = 0.75
    v_fac = 1 - t_fac
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    train_dir = os.path.join(out_dir, "train")
    valid_dir = os.path.join(out_dir, "valid")
    os.mkdir(train_dir)
    os.mkdir(valid_dir)

    index_t = 0
    index_v = 0
    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        dirnames = np.random.permutation(dirnames)
        si = int(len(dirnames) * t_fac)
        for dir_name in dirnames:
            if index_t < len(dirnames) * t_fac:
                new_dir = os.path.join(train_dir, dir_name)
                os.mkdir(new_dir)
                dir_name = os.path.join(in_dir, dir_name)

                filenames = [f for f in os.listdir(os.path.abspath(dir_name))]
                for file in filenames:
                    if file.__contains__(".png"):
                        t = get_time(file)
                        if t < 50:
                            # new_name = "im" + str(t) + ".png"
                            new_name = os.path.basename(file)
                            new_path = os.path.join(new_dir, new_name)
                            shutil.copy(os.path.join(dir_name, file), new_path)
                index_t = index_t + 1
            else:
                new_dir = os.path.join(valid_dir, dir_name)
                os.mkdir(new_dir)
                dir_name = os.path.join(in_dir, dir_name)

                filenames = [f for f in os.listdir(os.path.abspath(dir_name))]
                for file in filenames:
                    if file.__contains__(".png"):
                        t = get_time(file)
                        if t < 50:
                            # new_name = "im" + str(t) + ".png"
                            new_name = file
                            new_path = os.path.join(new_dir, new_name)
                            shutil.copy(os.path.join(dir_name, file), new_path)

                index_v = index_v + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs tests with varying input sizes.')
    parser.add_argument('-f',
                        dest='f',
                        help='Path to the directory containing the runs.')
    parser.add_argument('-out_dir',
                        dest='out_dir',
                        help='Path to the output photos.')
    parser.add_argument('-valid',
                        dest='v',
                        default=False,
                        help='Path to the output photos.')
    args = parser.parse_args()
    main(os.path.abspath(args.f), os.path.abspath(args.out_dir))
