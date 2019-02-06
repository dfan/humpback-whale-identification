import PIL
from PIL import Image
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tqdm import tqdm

def make_bbox_image(img_path):
  """
  :param img: path to image
  """
  main_img = Image.open(img_path)
  r_img = main_img.resize((128, 128), PIL.Image.ANTIALIAS)
  # convert to 1d image
  rb_img = r_img.convert('L')
  #rb_img_arr = np.asarray(rb_img, dtype=float)
  rb_img_arr = image.img_to_array(rb_img)
  x0, y0, x1, y1 = model.predict(np.expand_dims(rb_img_arr, axis=0)).squeeze()
  t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  trans = center_transform(t, rb_img_arr.shape)
  (u0, v0),(u1, v1) = coord_transform([(x0,y0),(x1,y1)], trans)
  width = height = 128
  bbox = [max(u0,0), max(v0,0), min(u1,width), min(v1,height)]
  if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
    bbox = [0,0,width,height]

  img_crop = r_img.crop(bbox)
  img_arr = image.img_to_array(img_crop)
  return img_crop

def coord_transform(list, trans):
  result = []
  for x,y in list:
    y,x,_ = trans.dot([y,x,1]).astype(np.int)
    result.append((x,y))
  return result

def center_transform(affine, input_shape):
  hi, wi = float(input_shape[0]), float(input_shape[1])
  ho, wo = float(img_shape[0]), float(img_shape[1])
  top, left, bottom, right = 0, 0, hi, wi
  if wi/hi/anisotropy < wo/ho: # input image too narrow, extend width
    w     = hi*wo/ho*anisotropy
    left  = (wi-w)/2
    right = left + w
  else: # input image too wide, extend height
    h      = wi*ho/wo/anisotropy
    top    = (hi-h)/2
    bottom = top + h
  center_matrix   = np.array([[1, 0, -ho/2], [0, 1, -wo/2], [0, 0, 1]])
  scale_matrix    = np.array([[(bottom - top)/ho, 0, 0], [0, (right - left)/wo, 0], [0, 0, 1]])
  decenter_matrix = np.array([[1, 0, hi/2], [0, 1, wi/2], [0, 0, 1]])
  return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))

# Define useful constants
img_shape  = (128,128,1)
anisotropy = 2.15
train_read_path = os.path.abspath('./data/train')
test_read_path = os.path.abspath('./data/test')
train_write_path = os.path.abspath('./data/train_cropped')
test_write_path = os.path.abspath('./data/test_cropped')
model = load_model(os.path.join('cropping.model'))

if not os.path.exists(train_write_path):
  os.makedirs(train_write_path)
if not os.path.exists(test_write_path):
  os.makedirs(test_write_path)

for img_name in tqdm(os.listdir(train_read_path)):
  read_img_path = os.path.join(train_read_path, img_name)
  cropped_img = make_bbox_image(read_img_path)
  write_img_path = os.path.join(train_write_path, img_name)
  cropped_img.save(write_img_path, 'JPEG')
"""
for img_name in tqdm(os.listdir(test_read_path)):
  read_img_path = os.path.join(test_read_path, img_name)
  cropped_img = make_bbox_image(read_img_path)
  write_img_path = os.path.join(test_write_path, img_name)
  cropped_img.save(write_img_path, 'JPEG')
"""
