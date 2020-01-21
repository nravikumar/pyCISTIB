import argparse
import os, glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from resnet3d import BuildCNN
from Utils import Generate_plots
from Data_gen import Datagen

def get_image_paths(base_path, apex_path, full_path):
    # Get image file paths
    num_pat = 10
    num_train = 8

    print('Reading dicom images...')
    folder_pattern = ''.join('/image/time0*/SAX/')

    folder_list = []
    train_paths=[]
    val_paths=[]
    all_pat_folders = os.listdir(base_path)
    for idx, pid in enumerate(all_pat_folders):
        if idx<num_pat:
            tmp_folder = ''.join(pid) + folder_pattern
            folder_list.append(sorted(glob.glob(base_path + tmp_folder)))

    print("Number of patient folders = ", len(folder_list))
    train_labels=[]
    val_labels=[]
    for idx, folder in enumerate(folder_list):
        print("Patient number: ", idx)
        for pidx, pat_folder in enumerate(folder):
            if idx==0 and pidx == 0:
                print(pat_folder)
            if idx < num_train:
                train_paths.append(pat_folder)
                try:
                    train_labels.append(1)
                except:
                    continue
            elif num_train <= idx < num_pat:
                try:
                    val_paths.append(pat_folder)
                    val_labels.append(1)
                except:
                    continue

    print("Number of training images currently = ", len(train_labels))
    print("Number of validation images currently = ", len(val_labels))
    print('Baseless images done..')

    all_pat_folders = os.listdir(apex_path)
    for idx, pid in enumerate(all_pat_folders):
        if idx<num_pat:
            tmp_folder = ''.join(pid) + folder_pattern
            folder_list.append(sorted(glob.glob(apex_path + tmp_folder)))

    print("Number of patient folders = ", len(folder_list))
    for idx, folder in enumerate(folder_list):
        print("Patient number: ", idx)
        for pidx, pat_folder in enumerate(folder):
            if idx==0 and pidx == 0:
                print(pat_folder)
            if idx < num_train:
                try:
                    train_paths.append(pat_folder)
                    train_labels.append(2)
                except:
                    continue
            elif num_train <= idx < num_pat:
                try:
                    val_paths.append(pat_folder)
                    val_labels.append(2)
                except:
                    continue

    print("Number of training images currently = ", len(train_labels))
    print("Number of validation images currently = ", len(val_labels))
    print('Apexless images done..')

    all_pat_folders = os.listdir(full_path)
    for idx, pid in enumerate(all_pat_folders):
        if idx<num_pat:
            tmp_folder = ''.join(pid) + folder_pattern
            folder_list.append(sorted(glob.glob(full_path + tmp_folder)))

    print("Number of patient folders = ", len(folder_list))
    for idx, folder in enumerate(folder_list):
        print("Patient number: ", idx)
        for pidx, pat_folder in enumerate(folder):
            if idx==0 and pidx == 0:
                print(pat_folder)
            if idx < num_train:
                try:
                    train_paths.append(pat_folder)
                    train_labels.append(0)
                except:
                    continue
            elif num_train <= idx < num_pat:
                try:
                    val_paths.append(pat_folder)
                    val_labels.append(0)
                except:
                    continue

    print("Number of training images currently = ", len(train_labels))
    print("Number of validation images currently = ", len(val_labels))
    print('Full coverage images done..')

    train_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)
    # print(train_paths)
    # print(val_paths)
    return train_paths, val_paths, train_labels, val_labels


# def get_input_image(batch_path):
#     reader = sitk.ImageSeriesReader()
#     dc_names = reader.GetGDCMSeriesFileNames(batch_path)
#     reader.SetFileNames(dc_names)
#     image = reader.Execute()
#     arr = sitk.GetArrayFromImage(image)
#     arr = arr.astype('float32')
#     arr -= np.mean(arr)
#     arr = arr / np.std(arr)
#     return arr
#
# def train_batch_dir_generator(train_paths,y_train):
#     tindx = np.arange(len(train_paths))
#     np.random.shuffle(tindx)
#     while True:
#         for t in tindx:
#             # select paths/files for current batch with shuffling
#             batch_path = train_paths[t]
#             try:
#                 batch_x = get_input_image(batch_path)
#                 batch_y = y_train[t]
#             except:
#                 continue
#             batch_x = np.expand_dims(batch_x, axis=0)
#             batch_x = np.expand_dims(batch_x, axis=4)
#             batch_y = np.expand_dims(batch_y, axis=0)
#             yield batch_x, batch_y
#
# def val_batch_dir_generator(val_paths,y_val):
#     tindx = np.arange(len(val_paths))
#     np.random.shuffle(tindx)
#     while True:
#         for t in tindx:
#             # select paths/files for current batch with shuffling
#             batch_path = val_paths[t]
#             try:
#                 batch_x = get_input_image(batch_path)
#                 batch_y = y_val[t]
#             except:
#                 continue
#             batch_x = np.expand_dims(batch_x, axis=0)
#             batch_x = np.expand_dims(batch_x, axis=4)
#             batch_y = np.expand_dims(batch_y, axis=0)
#             yield batch_x, batch_y


########################################################################################################################
# Main code snippet
########################################################################################################################


parser = argparse.ArgumentParser(description = 'Train CNN for classification.')
parser.add_argument('-b', metavar='Batch size', type=int, help='Specify batch size to use for training')
parser.add_argument('-n', metavar='Number of classes', type=int, help='Number of classes')
parser.add_argument('-e', metavar='Number of epochs', type=int, help='Number of epochs')
parser.add_argument('-d', metavar='CNN depth', type=int, help='Number of res blocks')
parser.add_argument('-l', metavar='Learning rate', type=float, help='Specify initial learning rate')
parser.add_argument('-c', metavar='Number of filters', type=int, help='Number of convolution kernels in first layer')

args = vars(parser.parse_args())

BATCHSIZE = args["b"]
EPOCHS = args["e"]
NUM_CLASSES = args["n"]

# base_path = "/MULTIX/DATA/INPUT/disk1/IQA_DATASET_WITHOUT_1_SLICE/BASELESS/"
# apex_path = "/MULTIX/DATA/INPUT/disk1/IQA_DATASET_WITHOUT_1_SLICE/APEXLESS/"
# # both_path = "/MULTIX/DATA/INPUT/disk3/IQA_DATASET_WITHOUT_3_SLICES/BOTHLESS/"
# full_path = "/MULTIX/DATA/INPUT/disk1/IQA_DATASET_WITHOUT_1_SLICE/FULL/"

base_path = "/usr/not-backed-up2/nishant/IQA/IQA_DATASET_WITHOUT_1_SLICE/Baseless/"
apex_path = "/usr/not-backed-up2/nishant/IQA/IQA_DATASET_WITHOUT_1_SLICE/Apexless"
full_path = "/usr/not-backed-up2/nishant/IQA/IQA_DATASET_WITHOUT_1_SLICE/Full"
train_paths, val_paths, y_train, y_val = get_image_paths(base_path, apex_path, full_path)

num_train = len(y_train)
print('Number of training samples = ', num_train)

# Convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
# print("Size of training labels = ", y_train.shape)

train_gen = Datagen.DataGenerator(y_train,folder_paths=train_paths,input_samples=None,num_classes=NUM_CLASSES,
                                  batch_size=BATCHSIZE,shuffle=True, RAM=False)
val_gen = Datagen.DataGenerator(y_val,folder_paths=val_paths,input_samples=None,num_classes=NUM_CLASSES,
                                batch_size=BATCHSIZE,shuffle=True,RAM=False)

# train_gen = train_batch_dir_generator(train_paths,y_train)
# val_gen = val_batch_dir_generator(val_paths,y_val)

img, label = next(train_gen)
tmp = np.squeeze(img[0,4,:,:,0])
plt.plot(tmp)
plt.imsave('ex_img',tmp,cmap='gray')

print("Example image shape = ", img.shape)
print("Training label = ", label)

data_aug = False
depth = args["d"]
cnn_model = BuildCNN.BuildNetwork(args["b"], args["n"], args["e"], y_train, y_val,
                         args["d"], data_aug, args["c"], depth, args["l"])

model = cnn_model.cnn_3d()
print(model.summary())

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '3DCNN_%s_model_{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_path = os.path.join(save_dir,model_name)

# Create model checkpoints
checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-12)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=1e-8,patience=10,mode='auto')
plot_losses = Generate_plots.PlotLosses()
callbacks = [checkpoint, lr_reducer, plot_losses, early_stopping]

print('####################### Starting training ########################')
history = model.fit_generator(generator=train_gen, steps_per_epoch=num_train//BATCHSIZE, epochs=EPOCHS, verbose=1,
                              callbacks=callbacks, validation_data=val_gen, validation_steps=len(y_val))

