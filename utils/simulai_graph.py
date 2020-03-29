
import json
import statistics
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def get_error(simul, real, k=1, mse=False):
    if mse:
        if k > 1:
            mse = [0] * len(simul[:])
            for i in range(len(simul[:])):
                for j in range(k):
                    mse[i] = mse[i] + (abs(simul[i][j]) - abs(real[j])) ** 2
                mse[i] = mse[i] / k
        else:
            mse = [0] * len(simul)
            for i in range(len(simul)):
                mse[i] = (abs(simul[i]) - abs(real)) ** 2
    else:
        if k > 1:
            mse = [0] * len(simul[:])
            # dev = [0] * len(simul[:])
            for i in range(len(simul[:])):
                for j in range(k):
                    # mse[i] = mse[i] + (abs(simul[i][j]) - abs(real[j])) ** 2
                    # dev[i] = dev[i] + (abs(real[j]) ** 2)
                    mse[i] = mse[i] + (abs(abs(simul[i][j]) - abs(real[j])) / abs(real[j]))
                # mse[i] = (mse[i] / dev[i]) ** 0.5

        else:
            mse = [0] * len(simul)
            dev = [0] * len(simul)
            for i in range(len(simul)):
                mse[i] = mse[i] + (abs(abs(simul[i]) - abs(real)) / abs(real))                # dev[i] = (abs(real) ** 2)
                # mse[i] = (mse[i] / dev[i]) ** 0.5
    return mse


def min_max_normalize(arr):
    return (arr - np.min(arr)) / np.ptp(arr)


def get_parameters(path):
    if not path.__contains__("="):
        path = os.path.normpath(path)
        path = "".join(path.split('/')[-1])
        tmp = path.split("_")
        for i in range(len(tmp)):
            if tmp[i].__contains__("gravity"):
                grav = tmp[i + 1]
            elif tmp[i].__contains__("amplitode"):
                amp = tmp[i + 1]
            elif tmp[i].__contains__("atwood"):
                at = tmp[i + 1]
            elif tmp[i].__contains__("time"):
                t1 = "".join(tmp[i + 1])
                t1 = t1.split('.png')[0]
    else:
        tmp = path.split("_")
        for i in range(len(tmp)):
            if tmp[i].__contains__("gravity"):
                grav = tmp[i + 1]
            elif tmp[i].__contains__("amplitode"):
                amp = tmp[i + 1]
            elif tmp[i].__contains__("atwood"):
                at = tmp[i + 1].split('/')[0]
        t1 = "".join(path.split('.png'))

        t1 = t1.split("time")[1]
        t1 = t1[1:]
    return grav, amp, at, t1


def plot_mse(file_dist1, info_rank, index, title, chisq=False):
    j = 1
    clr = ["or", "ok", "or", "ok", "or", "ok", "or", "ok", "or"]
    if not chisq:
        for file_dist in file_dist1:
            plt.figure(j)

            lowerlims = np.array([0] * len(info_rank))
            uplims = np.array([0] * len(info_rank))
            for i in range(len(info_rank)):
                if file_dist[i] > info_rank[i]:
                    lowerlims[i] = 0
                    uplims[i] = 1
                else:
                    lowerlims[i] = 1
                    uplims[i] = 0

            l2, caps, c2 = plt.errorbar(index, file_dist, lolims=lowerlims, \
                                        uplims=uplims, yerr=np.abs(np.subtract(file_dist, info_rank)),
                                        elinewidth=0.1,markeredgewidth=2,capsize=2, marker="o", ecolor="b", markersize=2, fmt=clr[j - 1])
            # index = np.array(index)
            # index = index[..., np.newaxis]
            # poly_fit = PolynomialFeatures(degree=(len(index)**0.5))
            # index_x = poly_fit.fit_transform(index)
            #
            # model = LinearRegression()
            # model.fit(index_x, file_dist)
            # y_poly_pred = model.predict(index_x)
            # plt.plot(index, y_poly_pred)
            # for k in range(len(file_dist)):
            #     chisq.append(stats.chisquare(file_dist[k], f_exp=info_rank[k])[0])
            # plt.plot(index, chisq)
            # plt.plot(index, file_dist)
            for cap in caps:
                cap.set_marker("o")
            # plt.plot(index, m, yerr=m + mse[1], color="or")
            j = j + 1
            plt.xlabel("Index of InfoGAN")
            plt.ylabel("Normalized rating")
    else:
        for file_dist in file_dist1:
            plt.figure(j)
            plt.plot(index, stats.chisquare(file_dist1, f_exp=info_rank)[1])
            # plt.errorbar(index, file_dist,  marker="o", ecolor="g", fmt=clr[j - 1])

            j = j + 1
            plt.xlabel("Index of InfoGAN")
            plt.ylabel("Normalized rating")
    plt.show()
    # plt.savefig(str(title) + ".png")


def combine_arrays(arr1, arr2, arr3=None, arr4=None):
    if arr3 is combine_arrays.__defaults__[0] and arr4 is combine_arrays.__defaults__[1]:
        ret = np.zeros(shape=(len(arr1), 2))
        for i in range(len(arr1)):
            ret[i][0] = arr1[i]
            ret[i][1] = arr2[i]
    elif arr4 is combine_arrays.__defaults__[1]:
        ret = np.zeros(shape=(len(arr1), 3))
        for i in range(len(arr1)):
            ret[i][0] = arr1[i]
            ret[i][1] = arr2[i]
            ret[i][2] = arr3[i]
    else:
        ret = np.zeros(shape=(len(arr1), 4))
        for i in range(len(arr1)):
            ret[i][0] = arr1[i]
            ret[i][1] = arr2[i]
            ret[i][2] = arr3[i]
            ret[i][3] = arr4[i]
    return ret


def combine_float(a, b, c=None, d=None):
    ret = []
    if c is combine_float.__defaults__[0] and d is combine_float.__defaults__[1]:
        ret.append(np.float64(a))
        ret.append(np.float64(b))
    elif d is combine_float.__defaults__[1]:
        ret.append(np.float64(a))
        ret.append(np.float64(b))
        ret.append(np.float64(c))
    else:
        ret.append(np.float64(a))
        ret.append(np.float64(b))
        ret.append(np.float64(c))
        ret.append(np.float64(d))
    return ret


def main(json_path):
    json_file = open(json_path, "r")
    data = json.load(json_file)
    num_test = 0
    gravity_median_show = []
    amp_median_show = []
    at_median_show = []
    gravity_amp_median_show = []
    gravity_at_median_show = []
    amp_at_median_show = []
    gravity_amp_at_median_show = []
    infogan_dist_median_show = []
    gravity_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    amplitude_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    atwood_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    dist_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    infogan_dist_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    gravity_amp_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    gravity_at_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    amp_at_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    gravity_amp_at_median = np.zeros(shape=(len(data["data"]), len(data["data"][0])))
    for comparison in data["data"]:
        print ("number test:", num_test)
        num_test = num_test + 1
        gravity = np.zeros(shape=(len(comparison)))
        amplitode = np.zeros(shape=(len(comparison)))
        atwood = np.zeros(shape=(len(comparison)))
        index = np.zeros(shape=(len(comparison)))
        time = np.zeros(shape=(len(comparison)))
        w = np.zeros(shape=(len(comparison)))
        title = []
        j = 0
        for i in range(len(comparison)):
            path = comparison[i]["path"]
            dist = comparison[i]["distance"]
            if dist == "0.0" or dist == "0" or dist == 0.0 or dist == 0:
                real_g, real_amp, real_at, real_time = get_parameters(path)
                real_g = np.float64(real_g)
                real_amp = np.float64(real_amp)
                real_at = np.float64(real_at)
                real_time = np.float64(real_time)
                real_path = path
            # else:
            w[j] = (comparison[i]["distance"])
            index[j] = np.int(comparison[i]["index"])
            g, amp, at, t = get_parameters(path)
            gravity[j] = (np.float64(g))
            amplitode[j] = (np.float64(amp))
            atwood[j] = (np.float64(at))
            time[j] = (np.float64(t))
            j = j + 1

        grav_amp_at_combined = combine_arrays(gravity, amplitode, atwood)
        grav_amp_combined = combine_arrays(gravity, amplitode)
        grav_at_combined = combine_arrays(gravity, atwood)
        amp_at_combined = combine_arrays(amplitode, atwood)
        real_grav_amp_at_combined = combine_float(real_g, real_amp, real_at)
        real_grav_amp_combined = combine_float(real_g, real_amp)
        real_grav_at_combined = combine_float(real_g, real_at)
        real_amp_at_combined = combine_float(real_amp, real_at)

        gravity_norm_err = min_max_normalize(get_error(gravity, real_g))
        amp_norm_err = min_max_normalize(get_error(amplitode, real_amp))
        at_norm_err = min_max_normalize(get_error(atwood, real_at))
        gravity_amp_norm_err = min_max_normalize(get_error(grav_amp_combined, real_grav_amp_combined, k=2))
        gravity_at_norm_err = min_max_normalize(get_error(grav_at_combined, real_grav_at_combined, k=2))
        amp_at_norm_err = min_max_normalize(get_error(amp_at_combined, real_amp_at_combined, k=2))
        gravity_amp_at_norm_err = min_max_normalize(get_error(grav_amp_at_combined,
                                      real_grav_amp_at_combined, k=3))

        # Median
        gravity_median[num_test - 1, :] = gravity_norm_err
        amplitude_median[num_test - 1, :] = amp_norm_err
        atwood_median[num_test - 1, :] = at_norm_err
        infogan_dist_median[num_test - 1, :] = w
        gravity_amp_median[num_test - 1, :] = gravity_amp_norm_err
        gravity_at_median[num_test - 1, :] = gravity_at_norm_err
        amp_at_median[num_test - 1, :] = amp_at_norm_err
        gravity_amp_at_median[num_test - 1, :] = gravity_amp_at_norm_err
        # Average
        if num_test != 1:
            infogan_dist_avg = np.add(w, infogan_dist_avg)
            gravity_avg = np.add(gravity_norm_err, gravity_avg)
            amp_avg = np.add(amp_norm_err, amp_avg)
            at_avg = np.add(at_norm_err, at_avg)
            gra_amp_avg = np.add(gravity_amp_norm_err, gra_amp_avg)
            gra_at_avg = np.add(gravity_at_norm_err, gra_at_avg)
            amp_at_avg = np.add(amp_at_norm_err, amp_at_avg)
            gra_amp_at_avg = np.add(gravity_amp_at_norm_err, gra_amp_at_avg)
        else:
            infogan_dist_avg = w
            gravity_avg = gravity_norm_err
            amp_avg = amp_norm_err
            at_avg = at_norm_err
            gra_amp_avg = gravity_amp_norm_err
            gra_at_avg = gravity_at_norm_err
            amp_at_avg = amp_at_norm_err
            gra_amp_at_avg = gravity_amp_at_norm_err

    mse = []
    for i in range(len(gravity_avg)):
        gravity_median_show.append(statistics.median(gravity_median[:, i]))
        amp_median_show.append(statistics.median(amplitude_median[:, i]))
        at_median_show.append(statistics.median(atwood_median[:, i]))
        gravity_amp_median_show.append(statistics.median(gravity_amp_median[:, i]))
        gravity_at_median_show.append(statistics.median(gravity_at_median[:, i]))
        amp_at_median_show.append(statistics.median(amp_at_median[:, i]))
        gravity_amp_at_median_show.append(statistics.median(gravity_amp_at_median[:, i]))
        infogan_dist_median_show.append(statistics.median(infogan_dist_median[:, i]))
    gravity_median_show = min_max_normalize(gravity_median_show)
    amp_median_show = min_max_normalize(amp_median_show)
    at_median_show = min_max_normalize(at_median_show)
    gravity_amp_median_show = min_max_normalize(gravity_amp_median_show)
    gravity_at_median_show = min_max_normalize(gravity_at_median_show)
    amp_at_median_show = min_max_normalize(amp_at_median_show)
    gravity_amp_at_median_show = min_max_normalize(gravity_amp_at_median_show)
    infogan_dist_median_show = min_max_normalize(infogan_dist_median_show)

    gravity_avg_show = min_max_normalize([x / num_test for x in gravity_avg])
    amp_avg_show = min_max_normalize([x / num_test for x in amp_avg])
    at_avg_show = min_max_normalize([x / num_test for x in at_avg])
    gra_amp_avg_show = min_max_normalize([x / num_test for x in gra_amp_avg])
    gra_at_avg_show = min_max_normalize([x / num_test for x in gra_at_avg])
    amp_at_avg_show = min_max_normalize([x / num_test for x in amp_at_avg])
    gra_amp_at_avg_show = min_max_normalize([x / num_test for x in gra_amp_at_avg])
    infogan_dist_avg_show = min_max_normalize([x / num_test for x in infogan_dist_avg])

    mse.append(gravity_avg_show)
    title.append("Compare parameter: {0}".format("Gravity - Average"))
    mse.append(amp_avg_show)
    title.append("Compare parameter: {0}".format("Amplitude - Average"))
    mse.append(at_avg_show)
    title.append("Compare parameter: {0}".format("Atwood - Average"))
    mse.append(gra_amp_avg_show)
    title.append("Compare parameter: {0}".format("Gravity, Amplitude - Average"))
    mse.append(gra_at_avg_show)
    title.append("Compare parameter: {0}".format("Gravity, Atwood - Average"))
    mse.append(amp_at_avg_show)
    title.append("Compare parameter: {0}".format("Amplitude, Atwood - Average"))
    mse.append(gra_amp_at_avg_show)
    title.append("Compare parameter: {0}".format("Gravity, Amplitude, Atwood - Average"))

    index = range(0, len(index))
    plot_mse(mse, infogan_dist_avg_show, index, title, False)

    mse = []
    title = []
    mse.append(gravity_median_show)
    title.append("Compare parameter: {0}".format("Gravity - Median"))
    mse.append(amp_median_show)
    title.append("Compare parameter: {0}".format("Amplitude - Median"))
    mse.append(at_median_show)
    title.append("Compare parameter: {0}".format("Atwood - Median"))
    mse.append(gravity_amp_median_show)
    title.append("Compare parameter: {0}".format("Gravity, Amplitude - Median"))
    mse.append(gravity_at_median_show)
    title.append("Compare parameter: {0}".format("Gravity, Atwood - Median"))
    mse.append(amp_at_median_show)
    title.append("Compare parameter: {0}".format("Amplitude, Atwood - Median"))
    mse.append(gravity_amp_at_median_show)
    title.append("Compare parameter: {0}".format("Gravity, Amplitude, Atwood - Median"))
    plot_mse(mse, infogan_dist_median_show, index, title, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs tests with varying input sizes.')
    parser.add_argument('-f',
                        dest='f',
                        help='Path to the directory containing the runs.')
    parser.add_argument('-r',
                        dest='real',
                        help='Path to the output photos.')
    args = parser.parse_args()
    main(os.path.abspath(args.f))
