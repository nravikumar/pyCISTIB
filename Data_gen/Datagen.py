import numpy as np
import keras
import SimpleITK as sitk
import numpy as np

class DataGenerator(keras.utils.Sequence):
    # Generate batches for network training from directory or RAM

    def __init__(self, labels, folder_paths=None, input_samples=None, num_classes=3, batch_size=1, shuffle=True, RAM=0):
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
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, idx):
        # Generate one batch of data
        current_indexes = self.current_indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        current_list_IDs = [self.list_IDs[k] for k in current_indexes]
        # Generate Data from RAM or hard-drive
        if self.RAM_gen == 1:
            batch_x, batch_y = self.__data_generator_RAM(current_list_IDs)
        else:
            batch_x, batch_y = self.__data_generator_HD(current_list_IDs)
        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle input samples each epoch
        self.current_indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.current_indexes)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


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
            batch_x[i] = np.expand_dims(batch_x[i], axis=0)
            batch_x[i] = np.expand_dims(batch_x[i], axis=4)
            batch_y[i] = np.expand_dims(batch_y[i], axis=0)
        return batch_x, batch_y

    def __data_generator_HD(self, current_list_IDs):
        # Generates batches with samples read from hard-drive
        batch_x=[]
        batch_y=[]
        fx = 15
        fz = 250
        for i, pidx in enumerate(current_list_IDs):
            folder_path = self.folder_paths[pidx]
            tmp_x = self.get_input_image(folder_path)
            tmp_y = self.labels[pidx]
            l_x = tmp_x.shape[0]
            pad_size = fx - l_x
            pad_array = np.zeros((pad_size, tmp_x.shape[1], tmp_x.shape[2]))
            tmp_x = np.concatenate((tmp_x,pad_array),axis=0)
            l_z = tmp_x.shape[2]
            pad_size = fz - l_z
            pad_array = np.zeros((tmp_x.shape[0], tmp_x.shape[1], pad_size))
            tmp_x = np.concatenate((tmp_x,pad_array),axis=2)
            # tmp_x = np.expand_dims(tmp_x, axis=0)
            tmp_x = np.expand_dims(tmp_x, axis=4)
            # tmp_y = np.expand_dims(tmp_y, axis=0)
            batch_x.append(tmp_x)
            batch_y.append(keras.utils.to_categorical(tmp_y, num_classes=self.num_classes))
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        # print("Current batch shape = ", batch_x.shape)
        return batch_x, batch_y


