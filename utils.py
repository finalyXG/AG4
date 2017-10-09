"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import copy
from time import gmtime, strftime

import os
import glob
from shutil import copyfile
import ipdb as pdb
import tensorflow as tf
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
	def __init__(self, maxsize=50):
		self.maxsize = maxsize
		self.num_img = 0
		self.images = []
	def __call__(self, image):
		if self.maxsize == 0:
			return image
		if self.num_img < self.maxsize:
			self.images.append(image)
			self.num_img=self.num_img+1
			return image
		if np.random.rand() > 0.5:
			idx = int(np.random.rand()*self.maxsize)
			tmp = copy.copy(self.images[idx])
			self.images[idx] = image
			return tmp
		else:
			return image

def load_test_data(image_path, fine_size=256):
	img = imread(image_path)
	img = scipy.misc.imresize(img, [fine_size, fine_size])
	img = img/127.5 - 1
	return img
def load_data_video(image_path, flip=True, is_test=False):

	img_list = load_video(image_path)
#	img_A, img_B, img_C, img_D, img_E = load_video(image_path)
	#img_list = preprocess_A_and_B_video(img_list, flip=flip, is_test=is_test) not necessary

	i = 0
	for img in img_list:
		img = img / 127.5 - 1.
		img_list[i] = img
		i += 1

	#img_A = img_A/127.5 - 1.
	#img_B = img_B/127.5 - 1.
	#img_C = img_C/127.5 - 1.
	#img_D = img_D/127.5 - 1.
	#img_E = img_E/127.5 - 1.

	#img_ABCDE = np.concatenate(([img_A], [img_B], [img_C], [img_D], [img_E]), axis=0)
	# img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
	return np.array(img_list)

def load_data_image(image_path, flip=True, is_test=False):
	img_M = load_image(image_path)
	#img_M = preprocess_M(img_M,flip=flip,is_test=is_test) not necessary
	img_M = img_M/127.5 - 1.
	return img_M


# def load_data_old(image_path, flip=True, is_test=False):
# 	img_A, img_B = load_image(image_path)
# 	img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

# 	img_A = img_A/127.5 - 1.
# 	img_B = img_B/127.5 - 1.

# 	img_AB = np.concatenate((img_A, img_B), axis=2)
# 	# img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
# 	return img_AB

def load_image(image_path):
	img = imread(image_path)
	return img

def load_video(image_path):

	img_len = len(image_path)
	img_list = []
	for tmp_path in image_path:
		tmp_img = imread(tmp_path[0], is_grayscale=False)
		img_list.append(tmp_img)


	#img_A = imread(image_path[0],is_grayscale=True)
	#img_B = imread(image_path[1],is_grayscale=True)
	#img_C = imread(image_path[2],is_grayscale=True)
	#img_D = imread(image_path[3],is_grayscale=True)
	#img_E = imread(image_path[4],is_grayscale=True)

	if len(np.array(img_list).shape) == 2:
		for img in img_list:
			img = np.dstack((img,img,img))
		
	#	img_A = np.dstack((img_A,img_A,img_A))
#		img_B = np.dstack((img_B,img_B,img_B))
#		img_C = np.dstack((img_C,img_C,img_C))
#		img_D = np.dstack((img_D,img_D,img_D))
#		img_E = np.dstack((img_E,img_E,img_E))
#	return img_A, img_B, img_C, img_D, img_E
	return img_list


def load_image_old(image_path):
	img_A = imread(image_path[0])
	img_B = imread(image_path[1])
	return img_A, img_B







def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)
        







def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]) 








glob_fine_size = 128
def preprocess_A_and_B_video(img_list, load_size=224, fine_size=glob_fine_size, flip=False, is_test=False):
	if is_test:
		i = 0
		for img in img_list:
			img = scipy.misc.imresize(img, [fine_size, fine_size])
			img_list[i] = img
			 
		#img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
		#img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
		#img_C = scipy.misc.imresize(img_C, [fine_size, fine_size])
		#img_D = scipy.misc.imresize(img_D, [fine_size, fine_size])
		#img_E = scipy.misc.imresize(img_E, [fine_size, fine_size])
	else:

		i = 0
		for img in img_list:
			img = scipy.misc.imresize(img, [fine_size, fine_size])
			img_list[i] = img
		#img_A = scipy.misc.imresize(img_A, [load_size, load_size])
		#img_B = scipy.misc.imresize(img_B, [load_size, load_size])
		#img_C = scipy.misc.imresize(img_C, [load_size, load_size])
		#img_D = scipy.misc.imresize(img_D, [load_size, load_size])
		#img_E = scipy.misc.imresize(img_E, [load_size, load_size])

		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		
		i = 0
		for img in img_list:
			img = img[h1:h1+fine_size, w1:w1+fine_size,:]
			img_list[i] = img
		#img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
		#img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]
		#img_C = img_C[h1:h1+fine_size, w1:w1+fine_size]
		#img_D = img_D[h1:h1+fine_size, w1:w1+fine_size]
		#img_E = img_E[h1:h1+fine_size, w1:w1+fine_size]


		#if flip and np.random.random() > 0.5:
#       	img_A = np.fliplr(img_A)
#   		img_B = np.fliplr(img_B)
#			img_C = np.fliplr(img_C)
#			img_D = np.fliplr(img_D)
#			img_E = np.fliplr(img_E)

	#return img_A, img_B,img_C,img_D,img_E
	return img_list








def preprocess_M(img_M, load_size=224, fine_size=glob_fine_size, flip=True, is_test=False):
	if is_test:
		img_M = scipy.misc.imresize(img_M, [fine_size, fine_size])
	else:
		img_M = scipy.misc.imresize(img_M, [load_size, load_size])

		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		img_M = img_M[h1:h1+fine_size, w1:w1+fine_size]

		if flip and np.random.random() > 0.5:
			img_M = np.fliplr(img_M)

	return img_M
def preprocess_A_and_B_old(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
	if is_test:
		img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
		img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
	else:
		img_A = scipy.misc.imresize(img_A, [load_size, load_size])
		img_B = scipy.misc.imresize(img_B, [load_size, load_size])

		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
		img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

		if flip and np.random.random() > 0.5:
			img_A = np.fliplr(img_A)
			img_B = np.fliplr(img_B)

	return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
	return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
	if (is_grayscale):
		tmp =  scipy.misc.imread(path, flatten = True).astype(np.float)
                tmp = scipy.misc.imresize(tmp,[glob_fine_size,glob_fine_size])
                return tmp
	else:
		tmp = scipy.misc.imread(path, mode='RGB').astype(np.float)
                tmp = scipy.misc.imresize(tmp,[glob_fine_size,glob_fine_size,3])
                return tmp

def merge_images(images, size):
	return inverse_transform(images)

def merge(images, size):
	return images[0][0]
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = image

	return img

def imsave(images, size, path):
	return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
				resize_h=64, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
          x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
	# npx : # of pixels width/height of image
	if is_crop:
		cropped_image = center_crop(image, npx, resize_w=resize_w)
	else:
		cropped_image = image
	return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
	return (np.array(images)+1.)/2.


def selected_best_anime_image():
	manga_feat_dir = glob.glob('../../training_data/0*/manga_illvec.npy')
	manga_feat_dir.sort()
	anime_feat_dir = glob.glob('../../training_data/0*/illvec_512.npy')
	anime_feat_dir.sort()
	manga_anime_dirs = zip(manga_feat_dir,anime_feat_dir)
	anime_imgs_folder_dir = glob.glob('../../training_data/*')
	anime_imgs_folder_dir.sort()
	idx = 0
	
	for manga_f, anime_f in manga_anime_dirs:
		np_manga = np.load(manga_f)
		np_animes = np.load(anime_f)
		np_dist = np.zeros((np_animes.shape[0],1))
		np_animes_diff = zip(np_animes,np_dist)
		
		i = 0
		for frame_f,dist in np_animes_diff:
			diff = frame_f - np_manga
			np_dist[i] = np.linalg.norm(diff)
			i += 1
		
		
		select_idx = np.argmin(np_dist)

		anime_images = glob.glob(anime_imgs_folder_dir[idx] + '/img_112_112_*.jpg' )
		anime_images.sort()
		
		image_selected_dir = anime_images[select_idx]
		
		copyfile(image_selected_dir, anime_imgs_folder_dir[idx] + '/selected_frames.jpg')
		   
		
		idx += 1
		if idx % 100== 0:
			print idx,
			
def video_summary(name,Video,fs):
	shape =tf.shape(Video)
	sum=[]
	for x in range(fs):
		frame = tf.slice(Video,[0,x,0,0,0],[-1,1,-1,-1,-1])
		frame = tf.reshape(frame,[shape[0],shape[2],shape[3],shape[4]])
		sum.append(tf.summary.image(name, frame))
	return sum