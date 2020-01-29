import keras
import keras.backend as K
from keras.layers import Input, Conv3D, GlobalMaxPooling3D, MaxPooling3D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.models import Model
from keras.optimizers import sgd, Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.utils.multi_gpu_utils import multi_gpu_model


class BuildNetwork(object):

    def __init__(self, batch_size,number_classes, epochs, train_labels, val_labels,
                 data_aug=False, num_filters=16, depth=3, lr=1e-3):

        self.batch_size = batch_size
        self.input_shape = (3,220,250,1)
        self.num_class = number_classes
        self.epochs = epochs
        self.num_filters = num_filters
        # self.train_data = train_data
        self.train_labels = train_labels
        # self.val_data = val_data
        self.val_labels = val_labels
        self.data_aug = data_aug
        self.depth = depth
        self.lr = lr
        self.kernel_size = (2,3,3)
        self.activation='relu'
        # self.activation = LeakyReLU(alpha=0.3)
        self.stride = 1
        self.num_conv_layers = 2

    def conv_layer(self, inputs):
        conv = Conv3D(self.num_filters, kernel_size=self.kernel_size, strides=self.stride,
                      padding='same', kernel_initializer=keras.initializers.he_normal(seed=7), kernel_regularizer=l2(1e-4))
        x = inputs
        x = conv(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(self.activation)(x)
        return x

    def dense_block(self, x):
        # Create dense conv block
        self.stride=1
        x = BuildNetwork.conv_layer(self, inputs=x)
        for l in range(self.num_conv_layers):
            if l == 0:
                y = BuildNetwork.conv_layer(self, inputs=x)
                cy = keras.layers.concatenate([y,x], axis=-1)
            else:
                y = BuildNetwork.conv_layer(self, inputs=cy)
                cy = keras.layers.concatenate([cy,y], axis=-1)
        x = MaxPooling3D(pool_size=(1,2,2))(cy)
        return x

    def cnn_3d(self):
        # Create CNN architecture
        self.curr_resblock=[]
        print('Initial learning rate = ', self.lr)
        inputs = Input(shape=self.input_shape)
        for res_block in range(self.depth):
            self.curr_resblock = res_block
            print('Creating DenseBlock: ', res_block)
            if res_block == 0:
                x = BuildNetwork.dense_block(self, inputs)
            else:
                x = BuildNetwork.dense_block(self, x)
            self.num_filters *= 2
        # gp = GlobalMaxPooling3D()(x)
        c1 = Conv3D(3,kernel_size=(1,1,1),strides=self.stride, padding='same',
                    kernel_initializer=keras.initializers.he_normal(seed=7), kernel_regularizer=l2(1e-4))(x)
        c1 = BatchNormalization(axis=-1)(c1)
        c1 = Activation(self.activation)(c1)

        f = Flatten()(c1)
        FC1 = Dense(32,activation=self.activation, kernel_initializer=keras.initializers.he_normal(seed=7))(f)
        #DP1 = Dropout(0.5)(FC1)
        FC2 = Dense(6,activation=self.activation, kernel_initializer=keras.initializers.he_normal(seed=7))(FC1)
        DP2 = Dropout(0.5)(FC2)
        outputs = Dense(self.num_class,activation='softmax')(DP2)
        cnn3d = Model(inputs=inputs, outputs=outputs)
        parallel_model = multi_gpu_model(cnn3d, gpus=8)
        # cnn3d.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr,beta_1=0.9,beta_2=0.999,epsilon=1e-6,
        #                                                               amsgrad=True), metrics=['accuracy'])
        parallel_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr,beta_1=0.9,beta_2=0.999,epsilon=1e-6,
                                                                      amsgrad=True), metrics=['accuracy'])
        # plot_model(cnn3d, to_file='CNN3D.png')
        plot_model(parallel_model, to_file='CNN3D.png')
        return parallel_model

