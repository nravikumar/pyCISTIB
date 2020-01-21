import keras
import keras.backend as K
from keras.layers import Input, Conv3D, GlobalMaxPooling3D, MaxPooling3D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Activation, Dropout
from keras.models import Model
from keras.optimizers import sgd, Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model



class BuildNetwork(object):

    def __init__(self, batch_size, number_classes, epochs, train_labels, val_labels,
                 cnn_dim, data_aug=False, num_filters=16, depth=3, lr=1e-3):
        self.batch_size = batch_size
        self.num_class = number_classes
        self.epochs = epochs
        self.num_filters = num_filters
        # self.train_data = train_data
        self.train_labels = train_labels
        # self.val_data = val_data
        self.val_labels = val_labels
        self.cnn = cnn_dim
        self.data_aug = data_aug
        self.depth = depth
        self.lr = lr

    def res_layer(self, inputs):
        conv = Conv2D(self.num_filters, kernel_size=self.kernel_size, strides=self.stride,
                      padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-2))
        x = inputs
        x = conv(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        return x

    def res_block(self, x):
        # Create residual conv block
        if self.curr_resblock > 1:
            self.stride = 2
        x = BuildNetwork.res_layer(self, inputs=x)
        self.stride=1
        for l in range(self.num_conv_layers):
            # print('Layer number = ', l)
            if l == 0:
                y = BuildNetwork.res_layer(self, inputs=x)
            else:
                y = BuildNetwork.res_layer(self, inputs=y)
        x = keras.layers.add([x,y])
        x = Activation(self.activation)(x)
        return x

    def cnn_2d(self):
        input_shape = self.train_data.shape[1:]
        # Create CNN architecture
        self.kernel_size = 3
        self.activation=LeakyReLU(alpha=0.3)
        self.stride = 1
        self.num_conv_layers = 2
        self.curr_resblock=[]
        inputs = Input(shape=input_shape)

        for res_block in range(self.depth):
            self.curr_resblock = res_block
            print('Creating ResBlock: ', res_block)
            if res_block == 0:
                x = BuildNetwork.res_block(self, inputs)
            else:
                x = BuildNetwork.res_block(self, x)
            self.num_filters *= 2
        gp = GlobalMaxPooling2D()(x)
        FC1 = Dense(6,activation=self.activation,kernel_initializer='he_normal')(gp)
        DP1 = Dropout(0.5)(FC1)
        outputs = Dense(self.num_class,activation='softmax',kernel_initializer='he_normal')(DP1)

        cnn2d = Model(inputs=inputs, outputs=outputs)
        cnn2d.compile(loss='categorical_crossentropy', optimizer=Adam(self.lr),metrics=['accuracy'])
        plot_model(cnn2d, to_file='CNN2D.png')
        return cnn2d

    def res_layer_3d(self, inputs):
        conv = Conv3D(self.num_filters, kernel_size=self.kernel_size, strides=self.stride,
                      padding='same', kernel_initializer=keras.initializers.he_normal(seed=7), kernel_regularizer=l2(1e-4))
        x = inputs
        x = conv(x)
        x = BatchNormalization(axis=4)(x)
        x = Activation(self.activation)(x)
        return x

    def res_block_3d(self, x):
        # Create residual conv block
        # if self.curr_resblock > 0:
        #     self.stride = 3
        self.stride=1
        x = BuildNetwork.res_layer_3d(self, inputs=x)
        for l in range(self.num_conv_layers):
            # print('Layer number = ', l)
            if l == 0:
                y = BuildNetwork.res_layer_3d(self, inputs=x)
            else:
                y = BuildNetwork.res_layer_3d(self, inputs=y)
        x = keras.layers.add([x,y])
        # x = Activation(self.activation)(x)
        #x = MaxPooling3D(pool_size=(1,2,2))(x)
        return x

    def cnn_3d(self):
        # input_shape = self.train_data.shape[1:]
        # Create CNN architecture
        self.kernel_size = 3
        self.activation='relu'
        # self.activation = LeakyReLU(alpha=0.3)
        self.stride = 1
        self.num_conv_layers = 2
        self.curr_resblock=[]
        inputs = Input(shape=(None,None,None,1))

        # conv_input = Conv3D(filters=8, kernel_size=7, strides=self.stride,
        #               padding='same', kernel_initializer=keras.initializers.he_normal(seed=7), kernel_regularizer=l2(1e-4))
        # c1 = conv_input(inputs)
        # c1 = BatchNormalization(axis=4)(c1)
        # c1 = Activation(self.activation)(c1)

        for res_block in range(self.depth):
            self.curr_resblock = res_block
            print('Creating ResBlock: ', res_block)
            if res_block == 0:
                x = BuildNetwork.res_block_3d(self, inputs)
            else:
                x = BuildNetwork.res_block_3d(self, x)
            self.num_filters *= 2
        gp = GlobalMaxPooling3D()(x)
        FC1 = Dense(6,activation=self.activation, kernel_initializer=keras.initializers.he_normal(seed=7))(gp)
        DP1 = Dropout(0.5)(FC1)
        outputs = Dense(self.num_class,activation='softmax')(DP1)
        cnn3d = Model(inputs=inputs, outputs=outputs)
        cnn3d.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr,beta_1=0.9,beta_2=0.999,epsilon=1e-6,
                                                                      amsgrad=True), metrics=['accuracy'])
        plot_model(cnn3d, to_file='CNN3D.png')
        return cnn3d