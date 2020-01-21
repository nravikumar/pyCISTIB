import numpy as np
import keras
import SimpleITK as sitk


class DataGenerator(keras.utils.Sequence):
    # Generate batches for network training from directory or RAM

    def __init__(self, labels, folder_paths=None, input_samples=None, num_classes=3, batch_size=1, shuffle=True, RAM=False):
        self.batch_size=batch_size
        if folder_paths is not None:
            self.list_IDs = np.arange(len(folder_paths))
        else:
            self.list_IDs = np.arange(len(input_samples))
        self.labels = labels
        self.RAM_gen = RAM
        self.shuffle = shuffle
        self.folder_paths = folder_paths
        self.input_samples = input_samples
        self.num_classes = num_classes
    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, idx):
        # Generate one batch of data
        current_indexes = self.current_indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        current_list_IDs = [self.list_IDs[k] for k in current_indexes]

        # Generate Data from RAM or hard-drive
        if self.RAM_gen == True:
            batch_x, batch_y = self.__data_generator_RAM(current_list_IDs)
        else:
            batch_x, batch_y = self.__data_generator_HD(current_list_IDs)

    def on_epoch_end(self):
        # Shuffle input samples each epoch
        self.current_indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.current_indexes)

    def get_input_image(self, batch_path):
        reader = sitk.ImageSeriesReader()
        dc_names = reader.GetGDCMSeriesFileNames(batch_path)
        reader.SetFileNames(dc_names)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)
        arr = arr.astype('float32')
        arr -= np.mean(arr)
        arr = arr / np.std(arr)
        return arr

    def __data_generator_RAM(self, current_list_IDs):
        # Generates batches with samples stored in RAM
        batch_x=[]
        batch_y=[]
        for i, pidx in enumerate(current_list_IDs):
            batch_x[i,:] = self.input_samples[pidx]
            batch_y[i] = self.labels[i]
            batch_y[i] = keras.utils.to_categorical(batch_y[i], num_classes=self.num_classes)
            batch_x = np.expand_dims(batch_x, axis=0)
            batch_x = np.expand_dims(batch_x, axis=4)
            batch_y = np.expand_dims(batch_y, axis=0)
        return batch_x, batch_y

    def __data_generator_HD(self, current_list_IDs):
        # Generates batches with samples read from hard-drive
        batch_x=[]
        batch_y=[]
        for i, pidx in enumerate(current_list_IDs):
            folder_path = self.folder_paths[pidx]
            batch_x[i,:] = self.get_input_image(folder_path)
            batch_y[i] = self.labels[i]
            batch_y[i] = keras.utils.to_categorical(batch_y[i], num_classes=self.num_classes)
            batch_x = np.expand_dims(batch_x, axis=0)
            batch_x = np.expand_dims(batch_x, axis=4)
            batch_y = np.expand_dims(batch_y, axis=0)
        return batch_x, batch_y
