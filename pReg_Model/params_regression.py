from time import time
import numpy as np
from absl import flags, app
import os
from glob import glob
import pandas as pd
from prediction import predict_one_image, show_similar_imgs_with_params
from data_prepare import min_max_normalization, create_train_valid_dataframe, create_dataframe, params_names_to_indices
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None, "Path for model without extension")
flags.DEFINE_boolean("train", False, "True if training is desired")
flags.DEFINE_boolean("predict", False, "True if prediction is desired")
flags.DEFINE_boolean("similarity", False, "True if parameters similarity is desired")
flags.DEFINE_string("params", "g, amp, at, time", "Predicted params")
flags.DEFINE_string("data", None, "data path")



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout, \
    Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2

import tensorflow as tf
import keras

# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.tensorflow_backend.set_session(sess)





def conv_predict(input_shape, out_num=4):
    input_img = Input(shape=input_shape)

    # h = 178, w = 87

    x = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform',
               kernel_regularizer=l2(0.001))(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    # h = 89, w = 43

    x = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform',
               kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    # h = 44, w = 21

    x = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform',
               kernel_regularizer=l2(0.001))(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(250, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(200, activation='relu')(x)
    result = Dense(out_num, activation='sigmoid')(x)
    # parameters_shape = (4)
    # result = Reshape(parameters_shape)(x)

    params_predict = Model(input_img, result)
    optimizer = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    params_predict.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return params_predict


# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.show()
    plt.savefig('acc_vs_epochs.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r')
    plt.title('Training and validation loss')
    plt.show()

    plt.savefig('loss_vs_epochs.png')
    plt.close()



def train(params_names, train_path, valid_path):
    HEIGHT = 178
    WIDTH = 87
    BATCH_SIZE = 256
    EPOCHS_NUM = 1000

    train_df, valid_df, train_min_max, valid_min_max = create_train_valid_dataframe(train_path, valid_path)

    print("Min & Max value from train_min_max:")
    for t in train_min_max:
        t.print_min_max()
    print("Min & Max value from valid_min_max:")
    for v in valid_min_max:
        v.print_min_max()

    # create generator
    datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.1)

    train_it = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col=params_names,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="raw",
        target_size=(178, 87),
        color_mode='grayscale')

    valid_it = datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='path',
        y_col=params_names,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="raw",
        target_size=(178, 87),
        color_mode='grayscale')

    if K.image_data_format() == 'channels_first':
        input_shape = (1, HEIGHT, WIDTH)
    else:
        input_shape = (HEIGHT, WIDTH, 1)

    model = conv_predict(input_shape, out_num=len(params_names))

    tbCallBack = TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit_generator(train_it, epochs=EPOCHS_NUM,
                                  validation_data=valid_it,
                                  validation_steps=8, steps_per_epoch=16, callbacks=[tbCallBack],
                                  use_multiprocessing=True)

    model.save(FLAGS.model_name + ".h5")

    # Write the net summary
    with open(FLAGS.model_name + '_summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Saved model to disk")

    plot_training(history)
    return model, train_min_max


def main(_):
    if FLAGS.model_name is None:
        print("Please provide a name for the model by providing --model_name=NAME without extension")
        return
    if FLAGS.data is None:
        print("Please provide a path for the training and validation data")
        return
    if FLAGS.data[-1] == '/':
        train_path = FLAGS.data[:-1] + '/train/*'
        valid_path = FLAGS.data[:-1] + '/valid/*'
    else:
        train_path = FLAGS.data + '/train/*'
        valid_path = FLAGS.data + '/valid/*'

    params_names = [s.strip() for s in FLAGS.params.split(',')]
    params_indices = params_names_to_indices(params_names)

    train_min_max = None
    if FLAGS.train:
        model, train_min_max = train(params_names, train_path, valid_path)

    if FLAGS.predict:
        if train_min_max is None:
            _, train_min_max= create_dataframe(train_path, None)

        model = tf.keras.models.load_model(FLAGS.model_name + ".h5")

        # Test certain images
        predict_one_image(model,
                          "data/train_dir/train/gravity_-600_amplitude_0.1_atwood_0.1_time_0.6.png",
                          train_min_max, param_indices=params_indices)

        predict_one_image(model,
                          "/home/matanr/TradeMarker/ExperimentSmooth/gravity_-740_amplitode_unknown_atwood_0.155/time=0.4.png",
                          train_min_max, param_indices=params_indices)

        predict_one_image(model,
                          "/home/matanr/TradeMarker/ExperimentSmooth/gravity_-740_amplitode_unknown_atwood_0.155/time=0.2.png",
                          train_min_max, param_indices=params_indices)
    if FLAGS.similarity:
        if train_min_max is None:
            _, min_max_norm_list= create_dataframe(train_path, None)

        model = tf.keras.models.load_model(FLAGS.model_name + ".h5")

        """ Provide the desired DB, from which some random images will be searched (using parameters regression), 
        against the entire DB """
        input_db = "smaller_db_4g_5amp_less_time/*"
        all_db = glob(input_db)
        random.shuffle(all_db)
        random_subset_db = all_db[:2000]
        # random_subset_db = all_db[:10]


        show_similar_imgs_with_params(model,
                                      random_subset_db,
                                      input_db,
                                      min_max_norm_list, write_to_json=True)






if __name__ == "__main__":
   app.run(main)
