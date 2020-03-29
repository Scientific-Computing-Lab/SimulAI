import shutil
import numpy as np
import argparse
import os


def get_parameters(path):
    tmp = path.split("_")
    for i in range(len(tmp)):
        if tmp[i].__contains__("gravity"):
            grav = tmp[i + 1]
        elif tmp[i].__contains__("amplitode"):
            amp= tmp[i + 1]
        elif tmp[i].__contains__("atwood"):
            at = tmp[i + 1].split('/')[0]
    t1 = "".join(path.split('.png'))

    t1 = t1.split("time")[1]
    t1 = t1[1:]
    return grav, amp, at, t1


def main(folder, out_dir):
    train_frac = 0.7
    list_files = list()
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for file in filenames:
            if file.__contains__(".png"):
                list_files += [os.path.join(dirpath, file)]

    # for file in tmp:
    #     g, amp, at, t = get_parameters(file)
    #     list_files.append("gravity_" + g + "_amplitude_" + amp + "_atwood_" + at + "_time_" + t + ".png")

    os.chdir(out_dir)
    num_file = len(list_files)
    random_file_list = np.random.permutation(list_files)
    random_train_list = random_file_list[0:int(num_file * train_frac)]
    random_valid_list = random_file_list[int(num_file * train_frac):]

    if os.path.exists("valid"):
        shutil.rmtree("valid")
    os.mkdir("valid")
    os.chdir("valid")
    prefix = os.path.join(out_dir, "valid")
    print (len(random_valid_list))
    for file1 in random_valid_list:
        g, amp, at, t = get_parameters(file1)
        file_name = "gravity_" + g + "_amplitude_" + amp + "_atwood_" + at + "_time_" + t + ".png"
        shutil.copyfile(file1, os.path.join(prefix, file_name))

    os.chdir(out_dir)
    if os.path.exists("train"):
        shutil.rmtree("train")
    os.mkdir("train")
    os.chdir("train")
    prefix = os.path.join(out_dir, "train")
    for file1 in random_train_list:
        g, amp, at, t = get_parameters(file1)
        file_name = "gravity_" + g + "_amplitude_" + amp + "_atwood_" + at + "_time_" + t + ".png"
        shutil.copyfile(file1, os.path.join(file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs tests with varying input sizes.')
    parser.add_argument('-f',
                        dest='f',
                        help='Path to the directory containing the runs.')
    parser.add_argument('-out_dir',
                        dest='out_dir',
                        help='Path to the output photos.')
    args = parser.parse_args()
    main(os.path.abspath(args.f), os.path.abspath(args.out_dir))
