import numpy as np
import pandas as pd 
import random as rn
import tensorflow as tfr
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color
from sklearn.metrics import classification_report
import os
import cv2

class CNN_FP:

    def __init__(self):

        self.seed = 0

        self.num_pixels=-1
        self.num_classes=-1 #init rand 

        np.random.seed(self.seed) 
        rn.seed(self.seed) # Setting the same seed for repeatability
        tfr.random.set_seed(self.seed)

        self.data_path = 'inputs/chest-xray-pneumonia/chest_xray/chest_xray/'

        self.train_path = self.data_path + 'train/'
        self.test_path = self.data_path + 'test/'
        self.val_path = self.data_path + 'val/'

        self.img_size = 200

        self.train_df=None
        self.test_df=None

        self.train=None
        self.test=None 

        self.X_train=None
        self.y_train=None
        self.X_test=None
        self.y_test=None
        self.X_train2=None
        self.X_test2=None
        self.y_train2=None
        self.y_test2=None

        self.input_shape = None 

        self.callbacks1=None
        self.callbacks2=None
        self.callbacks3=None
        self.callbacks4=None 

        self.y_pred=None
        self.y_pre_test=None #classes to save as compelx np arrays (n-Dim)

    def read_data(self, data_paths):
        for data_path in data_paths:
            labels = ['PNEUMONIA', 'NORMAL']
            images = []
            y = []
            for label in labels:
                curr_path = data_path + label
                for img in os.listdir(curr_path):
                    if ('DS' not in img):
                        image_path = os.path.join(curr_path, img)
                        image =  cv2.resize(cv2.imread(image_path), (self.img_size, self.img_size))
                        if image is not None:
                            images.append([image, label])
                    
        images = np.asarray(images)
        return images

    def set_train(self):

        self.train = self.read_data([self.train_path])
        self.test = self.read_data([self.val_path, self.test_path])

        for i in range(10):
            np.random.shuffle(self.train)
            np.random.shuffle(self.test)

        self.train_df = pd.DataFrame(self.train, columns=['image', 'label'])
        self.test_df = pd.DataFrame(self.test, columns = ['image', 'label'])

    def explore_plot(self): #explore 

        plt.figure(figsize=(18, 8))
        sns.set_style("darkgrid")

        plt.subplot(1,2,1)
        sns.countplot(self.train_df['label'], palette = 'coolwarm')
        plt.title('Train data')

        plt.subplot(1,2,2)
        sns.countplot(self.test_df['label'], palette = "hls")
        plt.title('Test data')

        plt.savefig("results/explore_cnn.png")

        pass

    def Show_example_image(self):

        fig = plt.figure(figsize = (16, 16))
        for idx in range(15):
            plt.subplot(5, 5, idx+1)
            plt.imshow(self.train_df.iloc[idx]['image'])
            plt.title("{}".format(self.train_df.iloc[idx]['label']))
            
        plt.tight_layout()

        plt.savefig("results/image_to_label_{}.png".format(np.random.uniform(0,1)))

        pass

    def splitdata(self, data):#data prep 
        X = []
        y = []
        for i, (val, label) in enumerate(data):
            X.append(val)
            y.append(self.lung_condition(label))
        return np.array(X), np.array(y)

    def call_splitdata(self):

        self.X_train, self.y_train = self.splitdata(self.train)
        self.X_test, self.y_test = self.splitdata(self.test)

        pass

    def preprocesing_to_cnn(self, data):
        data1 = color.rgb2gray(data).reshape(-1, self.img_size, self.img_size, 1).astype('float32')
        data1 /= 255
        return data1

    def call_pptocnn(self):

        self.X_train2 = self.preprocesing_to_cnn(self.X_train)
        self.X_test2 = self.preprocesing_to_cnn(self.X_test)

        self.y_train2 = to_categorical(self.y_train)
        self.y_test2 = to_categorical(self.y_test)

        self.num_classes = self.y_train2.shape[1]

        self.input_shape = (self.img_size, self.img_size, 1)

    def make_callbacks(self):

        self.callbacks1 = [EarlyStopping(monitor = 'loss', patience = 6), ReduceLROnPlateau(monitor = 'loss', patience = 3), ModelCheckpoint('models/model.best1.hdf5',monitor='loss', save_best_only=True)]
        self.callbacks3 = [EarlyStopping(monitor = 'loss', patience = 6), ReduceLROnPlateau(monitor = 'loss', patience = 3), ModelCheckpoint('models/model.best3.hdf5', monitor='loss' , save_best_only=True)]
        self.callbacks2 = [ EarlyStopping(monitor = 'loss', patience = 6), ReduceLROnPlateau(monitor = 'loss', patience = 3),  ModelCheckpoint('models/model.best2.hdf5', monitor='loss' , save_best_only=True)]
        self.callbacks4 = [EarlyStopping(monitor = 'loss', patience = 7), ReduceLROnPlateau(monitor = 'loss', patience = 4), ModelCheckpoint('models/model.best4.hdf5', monitor='loss' , save_best_only=True)]

        pass

    def lung_condition(self, label): #pre-proc
        if label == 'NORMAL':
            return 0
        else:
            return 1

    def set_preproc(self):

        np.random.shuffle(self.train)
        np.random.shuffle(self.test)
        self.X_train, self.y_train = self.splitdata(self.train) #overwrites 
        self.X_test, self.y_test = self.splitdata(self.test)

    def preprocesing_to_mlp(self, data):
        data1 = color.rgb2gray(data).reshape(-1, self.img_size * self.img_size).astype('float32')
        
        data1 /= 255 # Data Normalization [0, 1]
        
        return data1

    def mlp_tt(self): 

        self.X_train = self.preprocesing_to_mlp(self.X_train)
        self.X_test = self.preprocesing_to_mlp(self.X_test)

        pass

    def draw_learning_curve(self, history, keys=['accuracy', 'loss']):

        plt.figure(figsize=(20,8))
        for i, key in enumerate(keys):
            plt.subplot(1, 2, i + 1)
            sns.lineplot(x = history.epoch, y = history.history[key])
            sns.lineplot(x = history.epoch, y = history.history['val_' + key])
            plt.title('Learning Curve')
            plt.ylabel(key.title())
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='best')
        
        plt.savefig("results/learning_curve_nn_{}.png".format(np.random.uniform(0,1))) #random enough for nonoverlap 

        pass

    def get_mlp(self): #mlp layer, IO premade 
            
        return Sequential([Dense(1024, input_dim = self.num_pixels, activation='relu'), Dense(self.num_classes, activation='softmax')])

    def eval_model1(self):

        model = load_model('models/model.best1.hdf5')

        learning_history = model.fit(self.X_train, self.y_train, batch_size = 64, epochs = 40, verbose = 2, callbacks = self.callbacks1, validation_data=(self.X_test, self.y_test))

        self.draw_learning_curve(learning_history)

        score = model.evaluate(self.X_test, self.y_test, verbose = 0)
        print('Test loss: {}%'.format(score[0] * 100))
        print('Test accuracy: {}%'.format(score[1] * 100))

        print("MLP Error: %.2f%%" % (100 - score[1] * 100))

        pass

    def get_mlpv2(self):#model 2
            
        return Sequential([Dense(1024, input_dim=self.num_pixels, activation='relu'), Dropout(0.4), Dense(512, activation='relu'), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.3), Dense(self.num_classes, activation='softmax')])

    def pre_m2(self):

        model = self.get_mlpv2()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        print(model.summary())

        pass

    def eval_model2(self):

        model = load_model('models/model.best2.hdf5')

        learning_history = model.fit(self.X_train, self.y_train, batch_size = 64, epochs = 100, verbose = 1, callbacks = self.callbacks2, validation_data=(self.X_test, self.y_test))

        score = model.evaluate(self.X_test, self.y_test, verbose = 0)
        print('Test loss: {}%'.format(score[0] * 100))
        print('Test accuracy: {}%'.format(score[1] * 100))

        print("MLP Error: %.2f%%" % (100 - score[1] * 100))

        self.draw_learning_curve(learning_history)

        pass

    def data_aug(self): #data aug

        datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False, featurewise_std_normalization = False, samplewise_std_normalization = False, zca_whitening = False, horizontal_flip = False, vertical_flip = False, rotation_range = 10, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1)

        datagen.fit(self.X_train)
        self.train_gen = datagen.flow(self.X_train, self.y_train, batch_size = 32)

        pass

    def get_modelcnn(self): #CNN model

        return Sequential([Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape = self.input_shape), Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.25), Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.25), Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'), Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.25), Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ), Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.25), Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same' ), Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.25), Flatten(), Dense(512, activation='relu'), Dropout(0.5), Dense(256, activation='relu'), Dropout(0.5), Dense(64, activation='relu'), Dropout(0.5), Dense(self.num_classes, activation = "softmax")])

    def eval_cnn_m1(self): #fit and eval cnn 1

        model = self.get_modelcnn()
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print(model.summary())      

        learning_history = model.fit(self.X_train, self.y_train, batch_size = 64, epochs = 100, verbose = 1, callbacks = self.callbacks3, validation_data = (self.X_test, self.y_test))

        model = load_model('models/model.best3.hdf5')

        score = model.evaluate(self.X_test, self.y_test, verbose = 0)
        print('Test loss: {}%'.format(score[0] * 100))
        print('Test accuracy: {}%'.format(score[1] * 100))

        print("MLP Error: %.2f%%" % (100 - score[1] * 100))

        self.draw_learning_curve(learning_history)

        pass

    def get_modelcnn_v2(self): #cnn model 2:

        return Sequential([Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape = self.input_shape), Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.2), Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)),Dropout(0.2), Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'), Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.2), Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.2), Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'), Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPool2D(pool_size=(2, 2)), Dropout(0.2), Flatten(), Dense(1024, activation='relu'), BatchNormalization(), Dropout(0.5), Dense(512, activation='relu'), BatchNormalization(), Dropout(0.4), Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(64, activation='relu'), BatchNormalization(), Dropout(0.2), Dense(self.num_classes, activation = "softmax")])

    def eval_cnn_model2(self):

        model = self.get_modelcnn_v2()
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        model.summary()

        learning_history = model.fit_generator((self.train_gen), epochs = 100, steps_per_epoch = self.X_train.shape[0] // 32, validation_data = (self.X_test, self.y_test), callbacks = self.callbacks4)

        model = load_model('models/model.best4.hdf5')

        score = model.evaluate(self.X_test, self.y_test, verbose = 0)
        print('Test loss: {}%'.format(score[0] * 100))
        print('Test accuracy: {}%'.format(score[1] * 100))

        print("MLP Error: %.2f%%" % (100 - score[1] * 100))

        self.draw_learning_curve(learning_history)

        y_pred = model.predict(self.X_test) #pred 2
        self.y_pred = np.argmax(y_pred, axis = 1)

        self.y_pre_test = np.argmax(self.y_test, axis = 1)

        pass

    def show_condition(self, num):
        if num == 0:
            return 'NORMAL'
        return 'PNEUMONIA'

    def final_classification(self):

        cnt_error = []
        for idx, (a, b) in enumerate(zip(self.y_pre_test, self.y_pred)):
            if a == b: continue
            cnt_error.append(a)# test

        cnt_error = np.unique(cnt_error, return_counts = True)
        sns.set_style("darkgrid")
        plt.figure(figsize = (15, 7))
        sns.barplot([self.show_condition(x) for x in cnt_error[0]], cnt_error[1], palette="muted")
        plt.show()

        cnt_ind = 1
        list_idx = []
        fig = plt.figure(figsize=(14, 14))
        X_test_plot = self.X_test.reshape(-1, self.img_size, self.img_size)
        for idx, (a, b) in enumerate(zip(self.y_pre_test, self.y_pred)):
            if(cnt_ind > 16):break
            if a == b: continue
            plt.subplot(4, 4, cnt_ind)
            plt.imshow(X_test_plot[idx], cmap='gray', interpolation='none')
            plt.title('y_true = {0}\ny_pred = {1}\n ind = {2}'.format(self.show_condition(a), self.show_condition(b), idx))
            plt.tight_layout()
            list_idx.append(idx)
            cnt_ind += 1

        print(classification_report(self.y_pre_test, self.y_pred)) #report class

        pass

    def run_sequence_model1(self): #model 1 normal mlp

        ##code

        pass

    def run_sequence_model2(self): #model 2 normal mlp
    
        ##code

        pass

    def run_sequence_model3(self): #cnn layers v1
    
        ##code

        pass

    def run_sequence_model4(self): #cnn model v2
    
        ##code

        pass


if __name__=="__main__":

    cfp=CNN_FP()
    cfp.run_sequence_model1()
    cfp.run_sequence_model2()
    cfp.run_sequence_model3()
    cfp.run_sequence_model4()
