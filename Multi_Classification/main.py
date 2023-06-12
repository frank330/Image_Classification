# coding= utf-8
import os #用于读取文件路径
import numpy as np #用于数值操作
import cv2 # 用于图像处理
from tensorflow.python.keras.utils.np_utils import to_categorical # 将labels转成二进制形式
from tensorflow.python.keras.models import load_model # 加载模型
from classification_utilities import display_cm  # 用于给混淆矩阵加标签
from sklearn.model_selection import train_test_split #用于数据集的划分
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.python.keras.models import Model as keras_Model
from tensorflow.python.keras.callbacks import TensorBoard
from classification_utilities import display_cm
import sklearn.metrics as metrics


def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:
        return True
    else:
        return False


def read_name_list():
    name_list = []
    for child_dir in os.listdir("./dataset/"):
        name_list.append(child_dir)
    print(name_list)
    return name_list


# 建立一个用于存储和格式化读取训练数据的类
class DataSet(object):

    def __init__(self, path):
        self.size = 64
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.extract_data(path)
        # 在这个类初始化的过程中读取path下的训练数据

    # 输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
    # 返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)
    def read_file(self, path):
        img_list = []
        label_list = []
        dir_counter = 0
        n = 0
        # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            for dir_image in os.listdir(child_path):
                if endwith(dir_image, 'jpg'):
                    img = cv2.imread(os.path.join(child_path, dir_image))
                    img = cv2.resize(img, (self.size, self.size))
                    img_list.append(img)
                    label_list.append(dir_counter)
                    n = n + 1
            dir_counter += 1

        # 返回的img_list转成了 np.array的格式
        img_list = np.array(img_list)

        return img_list, label_list, dir_counter

    def extract_data(self, path):
        # 根据指定路径读取出图片、标签和类别数
        imgs, labels, counter = self.read_file(path)

        # 将数据集打乱随机分组
        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.3, )

        X_train = X_train.reshape(X_train.shape[0], self.size, self.size, 3)
        X_test = X_test.reshape(X_test.shape[0], self.size, self.size, 3)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        # 将labels转成 二进制形式
        Y_train = to_categorical(y_train, num_classes=counter)
        Y_test = to_categorical(y_test, num_classes=counter)

        # 将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def check(self):
        print('num of text dim:', self.X_test.ndim)
        print('text shape:', self.X_test.shape)
        print('text size:', self.X_test.size)
        print('num of train dim:', self.X_train.ndim)
        print('train shape:', self.X_train.shape)
        print('train size:', self.X_train.size)


class Model(object):
    FILE_PATH = r"./model/model.h5"

    def __init__(self):
        self.model = None

    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset

    def build_model(self):
        # self.base_model = InceptionV3(include_top=False)
        # self.base_model = VGG19()
        self.base_model = DenseNet201(include_top=False)
        # self.base_model = ResNet152(weights='imagenet')
        x = self.base_model.output
        # 加上一个全局平均池化层
        x = GlobalAveragePooling2D()(x)

        self.predictions = Dense(10, activation='softmax')(x)
        self.model = keras_Model(inputs=self.base_model.input, outputs=self.predictions)
        # 输出模型各层的参数状况
        self.model.summary()
        print('模型共有网络层数:' + str(len(self.model.layers)))

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    # 分类交叉熵损失，常搭配softmax用于多分类问题
    def train_model(self):
        self.model.compile(optimizer='Adamax',loss='categorical_crossentropy',metrics=['accuracy'])

        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=6, batch_size=20,
                       callbacks=[TensorBoard(log_dir='./log')]) # 使用tensorboard输出训练日志

    def evaluate_model(self):
        print('\nTesting---------------')

        facies_labels = read_name_list()

        y_pred_one = self.model.predict(self.dataset.X_test)  # shape=(n_samples, 10)
        y_pred_labels = np.argmax(y_pred_one,axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

        confusion_matrix = metrics.confusion_matrix(y_true=np.argmax(self.dataset.Y_test, axis=1),
                                                    y_pred=y_pred_labels)  # shape=(12, 12)

        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print('分类准确率 = %.4f' % accuracy)
        display_cm(confusion_matrix, facies_labels, hide_zeros=False)

    def save(self, file_path=FILE_PATH):
        self.model.save(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')

        self.model = load_model(file_path)


if __name__ == '__main__':
    datasets = DataSet(r"./dataset/")
    datasets.check()
    model = Model()
    model.read_trainData(datasets)
    model.build_model()
    # model.train_model()
    model.load()
    # model.save()
    model.evaluate_model()



