import keras
from keras import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dropout,BatchNormalization,MaxPool2D,GaussianNoise,ELU
from keras.layers import Dense,Activation
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.constraints import max_norm
from keras import regularizers
from keras.layers import Activation,UpSampling2D,Input,MaxPooling2D,Dropout
import numpy as np
from keras.constraints import max_norm
NUM_EPOCHS = 50
INIT_LR = 1e-4 * 0.3
BS = 20

class customLearningRate(keras.callbacks.Callback,LearningRateScheduler):
        def on_train_begin(self, logs={}):
                self.losses = []
        def on_epoch_begin(self,epoch,logs = {}):
                maxEpochs = NUM_EPOCHS
                baseLR = INIT_LR
                power = 0.5
                alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
                K.set_value(self.model.optimizer.lr,alpha)
                if epoch > 20:
                        if self.losses[epoch] < self.losses[epoch -5]:
                                lr_old = K.eval(self.model.optimizer.lr)
                                lr_new = lr_old * 2
                                INIT_LR = lr_new
                                K.set_value(self.model.optimizer.lr,lr_new)
                        
        def on_epoch_end(self,epoch,logs = {}):
                self.losses.append(logs.get('loss'))
custom = customLearningRate()
'''def poly_decay(epoch,lr):
        if epoch < 50:
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
                maxEpochs = NUM_EPOCHS
                baseLR = INIT_LR
                power = 0.5

	# compute the new learning rate based on polynomial decay
                alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
                return float(alpha)
        else:
                return float(lr + 0.00)'''
inp = Input(shape = (64,64,1))
x = Conv2D(32,kernel_size=(3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.))(inp)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = ELU()(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = ELU()(x)
x = Conv2D(32,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.))(x)
x1 = keras.layers.concatenate([x,inp])


for i in range(2):
    x = (Conv2D(64,kernel_size = (3,3,),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.)
                     ))(x)
    x = (BatchNormalization())(x)
    #x = Activation('relu')(x)
    x = ELU()(x)
    x = (MaxPool2D(pool_size = (2,2),strides = (2,2),padding = 'same'))(x)
ups = UpSampling2D((2,2))(x)
ups1 = UpSampling2D((2,2))(ups)
#ups1 = UpSampling2D((2,2))(ups1)
x = keras.layers.concatenate([ups1,x1])
x1 = Conv2D(64,kernel_size =(3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.))(x)
x = BatchNormalization()(x1)
#x = Activation('relu')(x)
x = ELU()(x)
#x = Dropout(0.6)(x)
x = Conv2D(128,kernel_size =(3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.))(x)
x1 = keras.layers.concatenate([x,x1])
x = BatchNormalization()(x1)
#x = Activation('relu')(x)
x = ELU()(x)
x  = Conv2D(128,kernel_size =(3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.))(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D((2,2),padding = 'same')(x)
x = ELU()(x)
x = Conv2D(256,kernel_size =(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_constraint = max_norm(3.))(x)
x = UpSampling2D()(x)
x1 = keras.layers.concatenate([x1,x])
x = BatchNormalization()(x1)
#x = Activation('relu')(x)
x = ELU()(x)
#x = Dropout(0.6)(x)
x = Conv2D(512,kernel_size =(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_constraint = max_norm(3.))(x)
#x = keras.layers.concatenate([x,x1])
x = Conv2D(256,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_constraint = max_norm(3.),activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(1024,kernel_constraint = max_norm(3.))(x)

x = BatchNormalization()(x)
#x1 = Activation('relu')(x)
x = ELU()(x)
#x  = BatchNormalization()(x1)
#x = Activation('relu')(x)
#x = Dense(64,kernel_constraint = max_norm(3.))(x1)
#x = Dropout(0.6)(x)
#x1 = keras.layers.concatenate([x1,x])

#x = Dense(32,kernel_constraint = max_norm(3.))(x)
#x = Dropout(0.6)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dense(16,kernel_constraint = max_norm(3.))(x)
x = Dropout(0.7)(x)
#x = keras.layers.concatenate([x,x1])
out = Dense(2,activation = 'softmax',kernel_constraint = max_norm(3.))(x)
model = Model(inp,out)

from skimage import data, img_as_float
from skimage import exposure

def preprocess(x):
        x = exposure.equalize_hist(x)
        x = x - tmean
        return x

train_datagen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             rotation_range = random.randint(1,10),
                             horizontal_flip = True,
                             vertical_flip = True,preprocessing_function = lambda x : x - tmean)
import os
from keras.preprocessing import image
path1 = "TrainDataset/train/DR"
path2 = "TrainDataset/train/NoDR"
listing1 = os.listdir(path1)
listing2 = os.listdir(path2)
all_images = []
for im in listing1:
            img = image.load_img("TrainDataset/train/DR/"+im, target_size=(64,64),color_mode = 'grayscale')
            x = image.img_to_array(img)
            all_images.append(x)
for im in listing2:
            img = image.load_img("TrainDataset/train/NoDR/"+im, target_size=(64,64),color_mode = 'grayscale')
            x = image.img_to_array(img)
            all_images.append(x)    
x_train = np.array(all_images)
tmean = np.mean(x_train, dtype=np.float64)
dev = np.std(x_train/255 - tmean)
print('deviation is ',dev)
train_datagen.fit(x_train,augment = True,rounds = 4)

train_generator = train_datagen.flow_from_directory('TrainDataset/train',
                                              target_size = (64,64),
                                              batch_size = 20,
                                              color_mode = 'grayscale',
                                              class_mode = 'categorical')

from keras.preprocessing import image
path11 = "TrainDataset/test/DR"
path22 = "TrainDataset/test/NoDR"
listing1 = os.listdir(path11)
listing2 = os.listdir(path22)
all_imagest = []
for im in listing1:
            img = image.load_img("TrainDataset/test/DR/"+im, target_size=(64,64),color_mode = 'grayscale')
            x = image.img_to_array(img)
            all_imagest.append(x)
for im in listing2:
            img = image.load_img("TrainDataset/test/NoDR/"+im, target_size=(64,64),color_mode = 'grayscale')
            x = image.img_to_array(img)
            all_imagest.append(x)
x_test = np.array(all_imagest)
test_datagen = ImageDataGenerator(rescale=1. / 255,preprocessing_function = lambda x : x - tmean)
test_datagen.fit(x_test,augment = True,rounds = 4)
validation_generator = test_datagen.flow_from_directory('TrainDataset/test',
                                                        target_size = (64,64),
                                                        batch_size = 20,
                                                        color_mode = 'grayscale',
                                                        class_mode = 'categorical')


#early_stop = EarlyStopping(monitor ='val_acc',patience = 5,verbose = 1,mode = 'auto',baseline = 0.5000,min_delta = 0.001,restore_best_weights = True)
early_stop = EarlyStopping(monitor ='val_acc',restore_best_weights = True)
#adam = keras.optimizers.Adam(lr = 0.0001,beta_1=0.9,beta_2 = 0.999,decay = 0.0001)
print(train_generator.class_indices)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
history = model.fit_generator(train_generator,epochs=110,steps_per_epoch=1058//20,verbose=1,validation_data = validation_generator, validation_steps = 357//20,callbacks = [custom])
#Saving the model
model.save('max_acc_res.h5')
#model.save_weights('max_acc_wt.h5')











