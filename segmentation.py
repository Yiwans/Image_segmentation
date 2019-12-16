import os
import sys
import random
import cv2
import tifffile as tiff
import tensorflow as tf
import numpy as np 
import pandas as pd 
from shapely.wkt import loads as wkt_loads

from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Dropout
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
import pydensecrf.densecrf as dcrf

N_CLASS = 10
INIT_DIR = '../input/'
IMG_SIZE = (1024, 1024)

def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
xf = W1 / Xmax
yf = H1 / Ymax
coords[:, 1] *= yf
coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
polygonList = None
if len(multipoly_def) > 0:
assert len(multipoly_def) == 1
polygonList = wkt_loads(multipoly_def.values[0])
return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
if polygonList is None:
return None
for k in range(len(polygonList)):
poly = polygonList[k]
perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
for pi in poly.interiors:
interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.uint8)
if contours is None:
return img_mask
    perim_list, interior_list = contours
cv2.fillPoly(img_mask, perim_list, class_value)
cv2.fillPoly(img_mask, interior_list, 0)
return img_mask

def _m(image_id):
filename = os.path.join(INIT_DIR, 'sixteen_band', '{}_M.tif'.format(image_id))
img = tiff.imread(filename)
return resize(img)


def _a(image_id):
filename = os.path.join(INIT_DIR, 'sixteen_band', '{}_A.tif'.format(image_id))
img = tiff.imread(filename)
return resize(img)


def resize(img, size=(IMG_SIZE[0], IMG_SIZE[1])):
c, w,h = np.shape(img)
    res_img = np.zeros([IMG_SIZE[0], IMG_SIZE[1], c])
for i in range(c):
        res_img[:,:,i] = cv2.resize(img[i,:,:].reshape((w,h,1)), IMG_SIZE)
return res_img


def stretch_n(bands, lower_percent=5, higher_percent=95):
out = np.zeros_like(bands)
    n = bands.shape[2]
for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
t[t < a] = a
t[t > b] = b
out[:, :, i] = t
return out.astype(np.float32)


classImage:
def __init__(self, M, A, ids, label=None):
        self.M = M
        self.A = A
if label is None:
label = np.zeros((IMG_SIZE[0], IMG_SIZE[1], N_CLASS))
        self.label = label
        self.ids = ids

imgs = []
for id_item in df.ImageId.unique():
    m = stretch_n(_m(id_item))
    a = stretch_n(_a(id_item))
img = Image(m, a, id_item)
xymax = _get_xmax_ymin(geo, id_item)
for i in range(10):
        polygon_list = _get_polygon_list(df, id_item, i)
contours = _get_and_convert_contours(polygon_list, IMG_SIZE, xymax)
img.label[:,:,i] = _plot_mask_from_contours(IMG_SIZE, contours, 1)
imgs.append(img)

classImgGenerator(Sequence):
def __init__(self, imgs, batch_size=32, size=128, train_flag=True):
        self.imgs = imgs
        self.batch_size = batch_size
        self.size = size
if train_flag:
right = 0
left = IMG_SIZE[0] - 2*size
else:
right = IMG_SIZE[0] - size
left = IMG_SIZE[0]

        self.right = right
        self.left = left
        self.train_flag = train_flag

def __len__(self):
return 1000000

def __getitem__(self, index):
        x_batch = []
        y_batch = []        
for _ in range(self.batch_size):
indx = random.randint(a=0, b=len(self.imgs)-1)
if self.train_flag:
ix = random.randint(a=self.right, b=self.left)
iy = random.randint(a=self.right, b=self.left)
else:
if random.uniform(0,1) > 0.5:
ix = random.randint(0, self.right)
iy = self.right
else:
iy = random.randint(0, self.right)
ix = self.right

xz = random.choice([-1,1])
yz = random.choice([-1,1])

#print(np.shape(self.imgs[indx]))
            img_m = self.imgs[indx].M[ix:ix+self.size, iy:iy+self.size, ::]
            img_a = self.imgs[indx].A[ix:ix+self.size, iy:iy+self.size, ::]
            x = np.dstack([img_m, img_a])
            y = self.imgs[indx].label[ix:ix+self.size, iy:iy+self.size, ::]
            x_batch.append(x)
            y_batch.append(y)
return np.array(x_batch), np.array(y_batch)

_smooth = 1e-12

def jaccard_coef(y_true, y_pred):
intersection = K.sum(y_true * y_pred, axis=[-3, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[-3, -1, -2])

jac = (intersection + _smooth) / (sum_ - intersection + _smooth)

return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

intersection = K.sum(y_true * y_pred_pos, axis=[-3, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[-3, -1, -2])
jac = (intersection + _smooth) / (sum_ - intersection + _smooth)
return K.mean(jac)

def dice_coef(y_true, y_pred, smooth=1):
intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
return (2. * intersection + smooth) / (K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + smooth)

def dice_coef_loss(y_true, y_pred):
return 1-dice_coef(y_true, y_pred)


def focal_loss(gamma=2, alpha=0.75):
def _focal_loss(y_true, y_pred):#with tensorflow
eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
return focal_loss

def get_unet():
inputs = Input((128,128, 16))
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.4)(pool3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.6)(pool4)

    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Convolution2D(N_CLASS, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs, conv10)
model.compile(optimizer= Adam(lr=1e-3, decay=1e-5), loss= binary_crossentropy, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
return model

def convolution(x, n, kernel_size, activation, padding):
    x = Convolution2D(n, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
return x


def get_fullconv():
inputs = Input((128,128, 16))
    conv1 = convolution(inputs, 32, (3, 3), activation='relu', padding='same')
    conv1 = convolution(conv1, 32, (3, 3), activation='relu', padding='same')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution(pool1,64, (3, 3), activation='relu', padding='same')
    conv2 = convolution(conv2,64, (3, 3), activation='relu', padding='same')
    conv2 = convolution(conv2,64, (3, 3), activation='relu', padding='same')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = convolution(pool2, 128, (3, 3), activation='relu', padding='same')
    conv3 = convolution(conv3, 128, (3, 3), activation='relu', padding='same')
    conv3 = convolution(conv3, 128, (3, 3), activation='relu', padding='same')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = convolution(pool3, 256, (3, 3), activation='relu', padding='same')
    conv4 = convolution(conv4, 256, (3, 3), activation='relu', padding='same')
    conv4 = convolution(conv4, 256, (3, 3), activation='relu', padding='same')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = convolution(pool4, 512, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 512, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 512, (3, 3), activation='relu', padding='same')
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv5 = convolution(pool5, 1024, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 1024, (3, 3), activation='relu', padding='same')

dr = Dropout(0.15)(conv5)
    conv10 = Convolution2D(N_CLASS, (1, 1), activation='sigmoid')(dr)

up = BilinearUpSampling2D(target_size=(128,128))(conv10)
model = Model(inputs, up)
model.compile(optimizer=Adam(lr=1*1e-3, decay=1e-5), loss=lambda y1,y2:binary_crossentropy(y1,y2), metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
return model

def get_deeplab():
inputs = Input((128,128, 16))
    conv1 = convolution(inputs, 32, (3, 3), activation='relu', padding='same')
    conv1 = convolution(conv1, 32, (3, 3), activation='relu', padding='same')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution(pool1,64, (3, 3), activation='relu', padding='same')
    conv2 = convolution(conv2,64, (3, 3), activation='relu', padding='same')
    conv2 = convolution(conv2,64, (3, 3), activation='relu', padding='same')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = convolution(pool2, 128, (3, 3), activation='relu', padding='same')
    conv3 = convolution(conv3, 128, (3, 3), activation='relu', padding='same')
    conv3 = convolution(conv3, 128, (3, 3), activation='relu', padding='same')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = convolution(pool3, 256, (3, 3), activation='relu', padding='same')
    conv4 = convolution(conv4, 256, (3, 3), activation='relu', padding='same')
    conv4 = convolution(conv4, 256, (3, 3), activation='relu', padding='same')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = convolution(pool4, 512, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 512, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 512, (3, 3), activation='relu', padding='same')

psp = []
for i in [1,3,7]:
artrous = Conv2D(512, (2,2), dilation_rate=(i,i),padding='same')(conv5)
artrous = BatchNormalization()(artrous)
artrous = Activation('relu')(artrous)
psp.append(artrous)
psp.append(conv5)
con = Concatenate()(psp)
dr = Dropout(0.25)(con)
    conv5 = convolution(dr, 1024, (3, 3), activation='relu', padding='same')
    conv5 = convolution(conv5, 1024, (3, 3), activation='relu', padding='same')

    conv10 = Convolution2D(N_CLASS, (1, 1), activation='sigmoid')(conv5)

up = BilinearUpSampling2D(target_size=(128,128))(conv10)
model = Model(inputs, up)
model.compile(optimizer= Adam(lr=1e-3, decay=1e-5), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
return model

def DownConv(inp):
def bottlenek(x_r, n_fm, kernel_size=3, strides=1, t=6, r=False):
 x = Conv2D(n_fm*t, 3, strides=strides, padding='same')(x_r)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

        x = Conv2D(n_fm*t, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
if r:
return Add()([x_r, x])
else:
            x = LeakyReLU(0.1)(x)
return x

# 128
    x = Conv2D(32, 3, strides=1, padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(32, 3, strides=1, padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

# 64
    x = bottlenek(x, 64, strides=1, t=1, r=False)
    x = bottlenek(x, 64, strides=1, t=1, r=True)
    x = bottlenek(x, 64, strides=1, t=1, r=True)
x1 = MaxPool2D(pool_size=2, strides=2)(x)

# 64
    x = bottlenek(x1, 128, strides=2, t=1, r=False)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
x2 = MaxPool2D(pool_size=2, strides=2)(x)

# 32
    x = bottlenek(x2, 128, strides=1, t=1, r=False)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
x3 = MaxPool2D(pool_size=2, strides=2)(x)

# 16
    x = bottlenek(x3, 128, strides=1, t=1, r=False)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
    x = bottlenek(x, 128, strides=1, t=1, r=True)
x4 = MaxPool2D(pool_size=2, strides=2)(x)
# 8
    x = bottlenek(x4, 256, strides=1, t=1, r=False)
    x = bottlenek(x, 256, strides=1, t=1, r=True)
    x = bottlenek(x, 256, strides=1, t=1, r=True)
x5 = MaxPool2D(pool_size=2, strides=2)(x)
# 4

return x1, x2, x3, x4, x5


def br_block(x_r, c, t=3):
    x = Conv2D(c*t, 3, strides=1, padding='same')(x_r)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(c*t, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

return Add()([x_r, x])


def gcn_block(x, c, t):
    x_left = Conv2D(c*t, [1, 3], strides=1, padding='same')(x)
    x_left = BatchNormalization()(x_left)
    x_left = LeakyReLU(0.1)(x_left)

    x_left = Conv2D(c*t, [3, 1], strides=1, padding='same')(x_left)
    x_left = BatchNormalization()(x_left)
    x_left = LeakyReLU(0.1)(x_left)

    x_right = Conv2D(c*t, [1, 3], strides=1, padding='same')(x)
    x_right = BatchNormalization()(x_right)
    x_right = LeakyReLU(0.1)(x_right)

    x_right = Conv2D(c*t, [3, 1], strides=1, padding='same')(x_right)
    x_right = BatchNormalization()(x_right)
    x_right = LeakyReLU(0.1)(x_right)
return Add()([x_left, x_right])


def get_model(inp_shape=(128,128,16), n_class = 10, activation='sigmoid'):

inp = Input(inp_shape)

x1, x2, x3, x4, x5 = DownConv(inp)

x2 = Dropout(0.05)(x2)
x3 = Dropout(0.1)(x3)
x4 = Dropout(0.2)(x4)
x5 = Dropout(0.3)(x5)


x2 = gcn_block(x2, c=n_class, t=1)
x3 = gcn_block(x3, c=n_class, t=1)
x4 = gcn_block(x4, c=n_class, t=1)
x5 = gcn_block(x5, c=n_class, t=1)

x2 = br_block(x2, c=n_class, t=1)
x3 = br_block(x3, c=n_class, t=1)
x4 = br_block(x4, c=n_class, t=1)
x5 = br_block(x5, c=n_class, t=1)

x5 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x5)

x4 = Add()([x5, x4])
x4 = br_block(x4, c=n_class, t=1)

x4 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x4)

x3 = Add()([x4, x3])
x3 = br_block(x3, c=n_class, t=1)

x3 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x3)

x2 = Add()([x3, x2])
x2 = br_block(x2, c=n_class, t=1)

x2 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x2)

x1 = br_block(x2, c=n_class, t=1)

x1 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x1)

x1 = br_block(x1, c=n_class, t=1)

x1 = Conv2DTranspose(n_class, 3, strides=2, padding='same')(x1)

x1 = br_block(x1, c=n_class, t=1)

out = Activation(activation)(x1)

model = Model(inputs=inp, outputs=out)

opt = Adam(lr=1e-3, decay=1e-5)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
print(model.summary())
return model

