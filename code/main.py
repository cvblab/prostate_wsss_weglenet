########################################################
# Imports
########################################################

import numpy
import os
from utils import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import cv2


# GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Seed for reproducibility
seed = 12
numpy.random.seed(seed)

########################################################
# Data directories and folders
########################################################

dir_data = '../data/images/'
dir_results = '../data/results/'
dir_partitions = '../data/partitions/'

########################################################
# Hyperparams
########################################################

optimizer = 'SGD'
learning_rate = 1*1e-3
batch_size = 8
n_epochs = 400
input_shape = (750, 750, 3)
aggregation = 'LSE'
r = 8
classes = ['NC', 'G3', 'G4', 'G5']

########################################################
# Dataset preparation
########################################################

data_frame = pd.read_excel(dir_partitions + 'partition.xlsx')

train = DataGenerator(data_frame=data_frame[data_frame['Partition'] == 'test'], path_images=dir_data,
                      input_shape=input_shape, batch_size=batch_size, data_augmentation=True,
                      shuffle=True, hide_and_seek=True)
val = DataGenerator(data_frame=data_frame[data_frame['Partition'] == 'val'], path_images=dir_data,
                    input_shape=input_shape, batch_size=batch_size, data_augmentation=False,
                    shuffle=False, hide_and_seek=False)
test = DataGenerator(data_frame=data_frame[data_frame['Partition'] == 'test'], path_images=dir_data,
                     input_shape=input_shape, batch_size=batch_size, data_augmentation=False,
                     shuffle=False, hide_and_seek=False)

########################################################
# WeGleNet training
########################################################

model = weglenet(input_shape=input_shape, output_shape=len(classes), aggregation=aggregation, learning_rate=1*1e-3,
                 freeze_up_to='', r=r)

checkpointer = ModelCheckpoint(filepath=dir_results + 'weights.hdf5', verbose=1, save_best_only=True)
lr_scheduler = LearningRateScheduler(scheduler)

H = model.fit_generator(train, epochs=n_epochs, steps_per_epoch=math.ceil(train.n / batch_size),
                        validation_data=val, validation_steps=math.ceil(val.n / batch_size),
                        callbacks=[checkpointer, lr_scheduler])
model.load_weights(dir_results + 'weights.hdf5')

# Learning curve plot
learning_curve_plot_generalized(H,  dir_results, 'lc', metrics=['binary_accuracy', 'val_binary_accuracy'],
                                losses=['loss', 'val_loss'])

########################################################
# Segmentation model
########################################################

model_segmentation = Model(inputs=model.input, outputs=model.get_layer('activation_1').output)
x = model_segmentation.get_layer('activation_1').output
# x = Activation('sigmoid')(x)
model_segmentation = Model(model_segmentation.input, x)
# Compile model
model_segmentation.summary()

########################################################
# Test segmentation masks prediction
########################################################

# Predict segmentation probabilty maps
y_pred_masks = model_segmentation.predict_generator(test, math.ceil(test.n / batch_size))
filenames = list(test.data_frame['image_name'])

# Save semantic segmentation masks
if not os.path.exists(dir_results + 'predictions_test/'):
    os.mkdir(dir_results + 'predictions_test/')

c = 0
for iFile in filenames:

    mask = cv2.resize(y_pred_masks[c, :, :, :], (3100, 3100))
    mask = np.argmax(mask, axis=2)

    cv2.imwrite(dir_results + 'predictions_test/' + iFile + '.png', mask)

    c += 1
print('Saved segmentation maps')