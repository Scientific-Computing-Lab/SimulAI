from glob import glob
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from data_prepare import min_max_normalization, create_dataframe
import os
import matplotlib.pyplot as plt
import cv2 as cv
import json
import pandas as pd



def load_image(img_path, height=178, width=87):

    img = image.load_img(img_path, target_size=(height, width), color_mode = "grayscale")
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def logit(x):
    return np.log(x / (1-x))


def predict_one_image(model, img_path, min_max_norm_list, param_indices=None, height=178, width=87):
    img = load_image(img_path, height, width)
    pred = model.predict(img)
    predicted_params = logit(pred[0, :])
    if param_indices is None:
        param_indices = range(len(predicted_params))
    for num_out, num_param in enumerate(param_indices):
        min_max_norm = min_max_norm_list[num_param]
        param = predicted_params[num_out]
        predicted_params[num_out] = (param * (min_max_norm.max_val - min_max_norm.min_val)) + min_max_norm.min_val

    print(predicted_params)
    return predicted_params


def l2_dist(params, row):
    row_params = [row['at'], row['time']]
    dist = np.linalg.norm(params - row_params)
    return dist


def show_similar_imgs_with_params(model, input_imgs_list, db_path, min_max_norm_list, height=178, width=87,
                                  write_to_json=True):
    data_json = {}
    data_json['data'] = []
    for counter, input_img in enumerate(input_imgs_list):
        if (counter + 1) % 10 == 0:
            print("Working on input number:{}".format(counter+1))
        img = load_image(input_img, height, width)
        pred = model.predict(img)
        df, min_max = create_dataframe(db_path, './all_df.csv')
        df['distance'] = df.apply(lambda x: l2_dist(pred[0, :], x), axis=1)
        sorted_distances = df.sort_values('distance', ascending=True)
        data = []
        data.insert(0, {'path': input_img, 'g': 0, 'amp': 0, 'at': 0, 'time': 0, 'distance': 0})
        sorted_distances = pd.concat([pd.DataFrame(data), sorted_distances], ignore_index=True, sort=False)
        sorted_distances['index'] = sorted_distances.index.values

        if not write_to_json:
            similarity_view(input_img, sorted_distances,
                            './params_similarity_out', min_max_norm_list)

        else:
            sub_data = sorted_distances[:2000]
            sub_data = sub_data[['path', 'distance', 'index']]

            sub_data_json = sub_data.to_json(orient='records', force_ascii=False)
            data_json['data'].append(sub_data_json)
    if write_to_json:
        with open('regression_results.json', 'w', encoding='utf-8') as json_file:
            json.dump(data_json, json_file, ensure_ascii=False)


def similarity_view(input_img, df, similarity_output, min_max_norm_list):
    if similarity_output is not None and not os.path.exists(similarity_output):
        os.makedirs(similarity_output)

    img_name = os.path.splitext(os.path.basename(input_img))[0]
    fig, axs = plt.subplots(nrows=3, ncols=6)
    fig.subplots_adjust(hspace=2)
    # for j in range(len(data_json[i])):
    new_path = os.path.join(similarity_output, img_name)
    os.makedirs(new_path)
    for i, ax in enumerate(fig.axes):
        if i == 0:
            img = cv.imread(input_img)
            ax.set_axis_off()
            ax.imshow(img)

            # time = str(img_name).split("=")
            # base_name = os.path.basename(os.path.dirname(str(input_img)))
            params = str(img_name).split("_")
            # print(params)
            name = "g:{},amp:{}\n,A:{},t:{}".format(params[1], params[3], params[5], params[7])

            cv.imwrite(new_path + "/input_" + img_name + ".png", img)
        else:
            path = df.loc[df.index[i], 'path']
            img = cv.imread(path)
            ax.set_axis_off()
            ax.imshow(img)

            params = np.array([df.loc[df.index[i], 'g'], df.loc[df.index[i], 'amp'], df.loc[df.index[i], 'at'], df.loc[df.index[i], 'time']])
            params = logit(params)
            for num_param, param in enumerate(params):
                min_max_norm = min_max_norm_list[num_param]
                params[num_param] = (param * (min_max_norm.max_val - min_max_norm.min_val)) + min_max_norm.min_val

            name = "g:{},amp:{}\n,A:{},t:{}".format(np.round(params[0], 2), np.round(params[1], 2), np.round(params[2], 2),
                                                    np.round(params[3], 2))

            out_img_name = os.path.splitext(os.path.basename(path))[0]
            cv.imwrite(new_path + "/" + str(i) + "_" + out_img_name + ".png", img)
        ax.set_title(str(name), pad=20)
    if similarity_output is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(20, 15)
        plt.savefig(similarity_output + "/" + img_name + ".png", bbox_inches='tight', dpi=fig.dpi)

    # plt.tight_layout()
    if similarity_output is None:
        plt.show()
