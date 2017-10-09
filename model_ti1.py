from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
import selectivesearch 
from module import *
from utils import *
#from g_model import GeneratorModel
#from d_model import DiscriminatorModel
import constants as c
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import misc
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error



class cyclegan(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.batch_size = args.batch_size
		self.image_size = args.fine_size
		self.frame_size = args.frame_size
		self.frames_nb = args.frames_nb
		self.input_c_dim = args.input_nc
		self.input_i_dim = args.input_ic
		self.output_c_dim = args.output_nc
		self.L1_lambda = args.L1_lambda
		self.dataset_dir = args.dataset_dir
		self.frames_nb = args.frames_nb


		self.discriminatorA = discriminator_video
		self.discriminatorB = discriminator_image
		if args.use_resnet:
			self.generatorA = generator_resnet_video
			self.generatorB = generator_resnet_image
		else:
			self.generatorA = None
			self.generatorB = None
		if args.use_lsgan:
			self.criterionGAN = mae_criterion
			#self.criterionGAN = abs_criterion
		else:
			self.criterionGAN = sce_criterion


		OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
							  gf_dim df_dim output_c_dim')
		self.options = OPTIONS._make((args.batch_size, args.fine_size,
									  args.ngf, args.ndf, args.output_nc))

		self._build_model()

		self.saver = tf.train.Saver()
		self.pool = ImagePool(args.max_size)

	def _build_model(self):
		# self.real_data = tf.placeholder(tf.float32,
		#                                 [None, self.image_size, self.image_size,
		#                                  self.input_c_dim + self.output_c_dim],
		#                                 name='real_A_and_B_images')
		self.real_data_video = tf.placeholder(tf.float32, 
										  [self.batch_size, self.frames_nb, self.frame_size, self.frame_size,
										   self.input_c_dim ],
										  name='real_videos')
		self.real_data_image = tf.placeholder(tf.float32,
											[self.batch_size, self.image_size, self.image_size,
										   self.input_i_dim],
										  name='real_images')
		


		self.real_A = self.real_data_video[:, :, :,:, :]
		self.real_B = self.real_data_image[:, :, :, :]
	   
		
		


		# <tf.Tensor 'generatorA2B/Tanh:0' shape=(1, 112, 112, 3) dtype=float32>    
		#self.fake_B = self.generatorB(self.real_A, self.options, False, name="generatorA2B")

		#<tf.Tensor 'generatorB2A/Tanh:0' shape=(1, 15, 112, 112, 3) dtype=float32>
		#self.fake_A_ = self.generatorA(self.fake_B, self.options, False, name="generatorB2A")

		#self.g_loss_a2b = self.L1_lambda * abs_criterion(self.fake_A_, self.fake_A_) 


		#<tf.Tensor 'generatorB2A/Tanh:0' shape=(1, 15, 112, 112, 3) dtype=float32>
		self.fake_A = self.generatorA(self.real_B, self.options, False, name="generatorB2A")


		#self.fake_B_ = self.generatorB(self.fake_A, self.options, True, name="generatorA2B")

		 #<tf.Tensor 'discriminatorA/Sigmoid:0' shape=(1, 1) dtype=float32>
		self.DA_fake = self.discriminatorA(self.fake_A,self.options, reuse=False, name="discriminatorA")


		#<tf.Tensor 'discriminatorB/d_h5_pred/Conv/convolution:0' shape=(1, 14, 14, 1) dtype=float32>
		#self.DB_fake = self.discriminatorB(self.fake_B,self.options, reuse=False, name="discriminatorB")


		#+ self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \  + self.L1_lambda * abs_criterion(self.real_A, self.fake_A) \
#		self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
#						  + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
#						  + self.L1_lambda * abs_criterion(self.real_B, self.fake_B)
		# print 1/0

		#  + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \ + self.L1_lambda * abs_criterion(self.real_B, self.fake_B)

				#train_op = tf.train.AdamOptimizer(1e-4).minimize(self.g_loss_b2a)
				#pdb.set_trace()	
				
		self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
						  + self.L1_lambda * abs_criterion(self.real_A, self.fake_A) 
#						  + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \

		#<tf.Tensor 'fake_A_sample:0' shape=(1, 15, 112, 112, 3) dtype=float32>      
		self.fake_A_sample = tf.placeholder(tf.float32,
											[self.batch_size, self.frames_nb, self.frame_size, self.frame_size,
											 self.input_c_dim], name='fake_A_sample')

		#<tf.Tensor 'fake_B_sample:0' shape=(1, 112, 112, 3) dtype=float32>
		self.fake_B_sample = tf.placeholder(tf.float32,
											[self.batch_size, self.image_size, self.image_size,
											 self.output_c_dim], name='fake_B_sample')


		# <tf.Tensor 'discriminatorB_1/d_h5_pred/Conv/convolution:0' shape=(1, 14, 14, 1) dtype=float32>
#		self.DB_real = self.discriminatorB(self.real_B, self.options, reuse=True, name="discriminatorB")


		#<tf.Tensor 'discriminatorA_1/Sigmoid:0' shape=(1, 1) dtype=float32>
		self.DA_real = self.discriminatorA(self.real_A, self.options, reuse=True, name="discriminatorA")

		#<tf.Tensor 'discriminatorB_2/d_h5_pred/Conv/convolution:0' shape=(1, 14, 14, 1) dtype=float32>
#		self.DB_fake_sample = self.discriminatorB(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")


		#<tf.Tensor 'discriminatorA_2/Sigmoid:0' shape=(1, 1) dtype=float32>
		self.DA_fake_sample = self.discriminatorA(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")


#		self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
#		self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
#		self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

		self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
		self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
		self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

#		self.g_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
		self.g_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
#		self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
		self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
#		self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
#		self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
		self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
		self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
#		self.db_sum = tf.summary.merge(
#			[self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum]
#		)

		self.da_sum = tf.summary.merge(
		   [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum]
		)

		#<tf.Tensor 'test_A:0' shape=(1, 15, 112, 112, 3) dtype=float32>
		self.test_A = tf.placeholder(tf.float32,
									 [self.batch_size, self.frames_nb, self.frame_size, self.frame_size,
									  self.input_c_dim], name='test_A')

		#<tf.Tensor 'test_B:0' shape=(1, 112, 112, 3) dtype=float32>
		self.test_B = tf.placeholder(tf.float32,
									 [self.batch_size, self.image_size, self.image_size,
									  self.input_i_dim], name='test_B')


		#<tf.Tensor 'generatorA2B_2/Tanh:0' shape=(1, 112, 112, 3) dtype=float32>
#		self.testB = self.generatorB(self.test_A, self.options, True, name="generatorA2B")
		
		#<tf.Tensor 'generatorB2A_2/Tanh:0' shape=(1, 15, 112, 112, 3) dtype=float32>
		self.testA = self.generatorA(self.test_B, self.options, True, name="generatorB2A")

		t_vars = tf.trainable_variables()
		self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
		self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
		self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
		self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
		for var in t_vars: print(var.name)

	def train(self, args):
		"""Train cyclegan"""

		self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			.minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)
				#print 1/0
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2
		self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			.minimize(self.da_loss, var_list=self.da_vars)

		#self.db_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
		#	.minimize(self.db_loss, var_list=self.db_vars)
		#self.g_a2b_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
		#	.minimize(self.g_loss_a2b, var_list=self.g_vars_a2b)

		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2
		self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			.minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
		init_op = tf.global_variables_initializer()

		self.sess.run(init_op)

		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

		counter = 1
		start_time = time.time()

		if self.load(args.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		#laurence 
		self.training_data_dir = '../training_data/'
		training_data_dir = self.training_data_dir
		M_dirs = np.load(training_data_dir + 'M_dirs.npy')
		A_dirs = np.load(training_data_dir + 'A_dirs.npy')

		for epoch in range(args.epoch):
				#print 1/0	

			# dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
			# dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))


			# np.random.shuffle(dataA)
			# np.random.shuffle(dataB)

			batch_idxs = min(min(M_dirs.shape[0], M_dirs.shape[0]), args.train_size) // self.batch_size
			self.train_size = args.train_size
			#idxx = np.random.randint(0,batch_idxs,batch_idxs)

			for idx in range(batch_idxs):

				
				A = A_dirs[idx]
				M0 = M_dirs[idx]
				M = []
				M = [M0[0][:6] + 'selected_frames.jpg']
				tmp_A = []
				for i in range(1,31,2):
					tmp_A.append(training_data_dir+A[i])

				A5 = list(zip(tmp_A))
				# A1 = list(zip([training_data_dir+A[0]],[training_data_dir+A[7]],[training_data_dir+A[14]],[training_data_dir+A[21]],[training_data_dir+A[28]]))
				m_image = load_data_image(training_data_dir + M[0],args)
				real_image = misc.imread(training_data_dir + M[0])
				a_video = load_data_video(A5,args)




				# img_lbl, regions = selectivesearch.selective_search(m_image, scale=10, sigma=0.2, min_size=10)
				# candidates = set()
				# for r in regions:
				# 	# excluding same rectangle (with different segments)
				# 	if r['rect'] in candidates:
				# 		continue
				# 	# excluding regions smaller than 2000 pixels
				# 	if r['size'] < 0.2000:
				# 		continue
				# 	# distorted rects
				# 	x, y, w, h = r['rect']
				# 	if w <= 0 or h <=0 or w / h > 1.2 or h / w > 1.2:
				# 		continue
				# 	candidates.add(r['rect'])
				# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
				# ax.imshow(m_image)
				# for x, y, w, h in candidates:
				# 	print x, y, w, h
				# 	rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
				# 	ax.add_patch(rect)

				# plt.show()

				#self.show_selective_search_rs(real_image,pscale=10, psigma=0.2, pmin_size=50)
				#regions = self.selective_search_rs(real_image,pscale=10, psigma=0.2, pmin_size=50)

				video_fh = a_video[0]
				video_ft = a_video[-1]

				#mask = np.zeros([self.image_size,self.image_size])

				#k = 1
				#for a_v_f in a_video[1::]:mask += abs(rgb2gray(a_v_f) - rgb2gray(video_fh))
				#mask[np.where(mask > 5)] = 0

				#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
				#ax.imshow(mask)
				#plt.show()
				#mask = np.reshape(mask,[self.image_size,self.image_size,1])
				#merge_image = np.concatenate([mask,m_image],axis=-1)
				#pdb.set_trace()
				merge_image = np.concatenate([video_fh,video_ft],axis=-1)



				#mask = self.get_mask(regions,video_fh,video_ft,0.5)
				#motion_area_loc = np.where(mask > 0)
				#motion_area_loc = zip(motion_area_loc[0],motion_area_loc[1])


				fake_A = self.sess.run([self.fake_A],feed_dict={self.real_B: [merge_image]})
				fake_A[0][0][0] = video_fh
				fake_A[0][0][-1] = video_ft

				#fake_A_comp = np.zeros([self.frames_nb,self.frame_size,self.frame_size,3])
				#for i in range(self.frames_nb):
				#	fake_A_comp[i] = fake_A[0][0][i]
				#	for jj,k in motion_area_loc:
				#		fake_A_comp[i,jj,k] = m_image[jj,k]

				#fake_A[0][0] = fake_A_comp


				
				#[fake_A, fake_B] = self.pool([fake_A, fake_B])
				#fake_B = self.pool(fake_B)
				#pdb.set_trace()

				# Update G network

				#_, summary_str = self.sess.run([self.g_a2b_optim, self.g_a2b_sum],							   feed_dict={self.real_A: [a_video],self.real_data_image:[m_image]})
				#self.writer.add_summary(summary_str, counter)
				# Update D network
				
				#_, summary_str = self.sess.run([self.db_optim, self.db_sum],feed_dict={self.real_B: [m_image], self.fake_B_sample: fake_B[0], self.real_data_image:[m_image]})

				#self.writer.add_summary(summary_str, counter)
				
				
				# Update G network
				_, summary_str = self.sess.run([self.g_b2a_optim, self.g_b2a_sum],
											   feed_dict={self.real_B: [merge_image],self.real_data_video:[a_video]})
				self.writer.add_summary(summary_str, counter)
				
				# Update D network
				_, summary_str = self.sess.run([self.da_optim, self.da_sum],feed_dict={self.real_A: [a_video],self.fake_A_sample: fake_A[0],self.real_data_video:[a_video]})

				self.writer.add_summary(summary_str, counter)

				counter += 1
				print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
					   % (epoch, idx, batch_idxs, time.time() - start_time)))
				if  counter == 2:
					self.sample_model(args.sample_dir, epoch, idx)

				if np.mod(counter, 100) == 1:
					self.sample_model(args.sample_dir, epoch, idx)

				if np.mod(counter, 1000) == 2:
					self.save(args.checkpoint_dir, counter)

	def get_mask(self,regions,video_fh,video_ft,th = 0.1):
		mask = np.zeros([self.image_size,self.image_size])

		for r in regions:
			crop_fh = video_fh[r[0]:r[0]+r[2],r[1]:r[1]+r[3],: ]
			crop_ft = video_ft[r[0]:r[0]+r[2],r[1]:r[1]+r[3],: ]
			ssim_noise = ssim(crop_fh, crop_ft, data_range=crop_fh.max() - crop_ft.min(),multichannel=True)
			mse_noise = mean_squared_error( rgb2gray(crop_fh), rgb2gray(crop_ft))  
			if ssim_noise < th and mse_noise < 80:
				mask[r[0]:r[0]+r[2],r[1]:r[1]+r[3]] = 1


		return mask


	def selective_search_rs(self,m_image, pscale=10, psigma=0.2, pmin_size=10):
		img_lbl, regions = selectivesearch.selective_search(m_image, scale=pscale, sigma=psigma, min_size=pmin_size)
		candidates = set()
		for r in regions:
			# excluding same rectangle (with different segments)
			if r['rect'] in candidates:
				continue
			# excluding regions smaller than 2000 pixels
			if r['size'] < 0.2000:
				continue
			# distorted rects
			x, y, w, h = r['rect']
			if w <= 10 or h <=10 or w / h > 1.5 or h / w > 1.5 or w > 60 or h > 60:
				continue
			candidates.add(r['rect'])
		return candidates
		


	def show_selective_search_rs(self,m_image, pscale=10, psigma=0.2, pmin_size=10,debug=False):
		img_lbl, regions = selectivesearch.selective_search(m_image, scale=pscale, sigma=psigma, min_size=pmin_size)
		candidates = set()
		for r in regions:
			# excluding same rectangle (with different segments)
			if r['rect'] in candidates:
				continue
			# excluding regions smaller than 2000 pixels
			if r['size'] < 0.2000:
				continue
			# distorted rects
			x, y, w, h = r['rect']
			if w <= 10 or h <=10 or w / h > 1.5 or h / w > 1.5 or w > 60 or h > 60:
				continue
			candidates.add(r['rect'])
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(m_image)
		for x, y, w, h in candidates:
			if debug:
				print x, y, w, h
			rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
			ax.add_patch(rect)

		plt.show()

	def save(self, checkpoint_dir, step):
		model_name = "cyclegan.model"
		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def sample_model(self, sample_dir, epoch, idx0):

		training_data_dir = self.training_data_dir
		M_dirs = np.load(self.training_data_dir + 'M_dirs.npy')
		A_dirs = np.load(self.training_data_dir + 'A_dirs.npy')

		

		
		batch_idxs = min(min(M_dirs.shape[0], M_dirs.shape[0]), self.train_size) // self.batch_size
		idx = np.random.randint(0,batch_idxs,3)
		for i in idx:
			A = A_dirs[i]
			M0 = M_dirs[i]
			M = []
			M = [M0[0][:6] + 'selected_frames.jpg']

			tmp_A = []
			for i in range(1,31,2):
				tmp_A.append(training_data_dir+A[i])

			A5 = list(zip(tmp_A))
			m_image = load_data_image(training_data_dir + M[0])
			real_image = misc.imread(training_data_dir + M[0])
			a_video = load_data_video(A5)


			#fake_B = self.sess.run([self.fake_B],feed_dict={self.real_A: [a_video]})
			#fake_A = self.sess.run(
			#	[self.fake_A],
			#	feed_dict={self.real_B:[m_image]}
			#	)
#

			#regions = self.selective_search_rs(real_image,pscale=10, psigma=0.2, pmin_size=50)

			video_fh = a_video[0]
			video_ft = a_video[-1]
			



			#mask = np.zeros([self.image_size,self.image_size])

			#k = 1
			#for a_v_f in a_video[1::]:mask += abs(rgb2gray(a_v_f) - rgb2gray(video_fh))
			#mask[np.where(mask > 5)] = 0

			#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
			#ax.imshow(mask)
			#plt.show()
			#mask = np.reshape(mask,[self.image_size,self.image_size,1])
			#merge_image = np.concatenate([mask,m_image],axis=-1)
			merge_image = np.concatenate([video_fh,video_ft],axis=-1)



			#mask = self.get_mask(regions,video_fh,video_ft,0.5)
			#motion_area_loc = np.where(mask > 0)
			#motion_area_loc = zip(motion_area_loc[0],motion_area_loc[1])


			fake_A = self.sess.run([self.fake_A],feed_dict={self.real_B: [merge_image]})
			fake_A[0][0][0] = video_fh
			fake_A[0][0][-1] = video_ft


			#fake_A_comp = np.zeros([self.frames_nb,self.frame_size,self.frame_size,3])
			#for i in range(self.frames_nb):
			#	fake_A_comp[i] = fake_A[0][0][i]
			#	for jj,k in motion_area_loc:
			#		fake_A_comp[i,jj,k] = m_image[jj,k]

			#fake_A[0][0] = fake_A_comp






						#save_images(fake_B, [self.batch_size, 1],'./{}/B_{}_{:03d}.jpg'.format(sample_dir, A[0][:5],epoch))
			tmp_dir = './{}/{}/'.format(sample_dir,epoch) 
			if not os.path.exists(tmp_dir):
				os.makedirs(tmp_dir)

			save_images([[ video_fh ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],14,epoch))
			save_images([[ video_ft ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],15,epoch))

			make_gif(a_video,'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(sample_dir,epoch,A[0][:5],16,epoch))
			make_gif(fake_A[0][0],'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(sample_dir,epoch,A[0][:5],17,epoch))


			#for j in range(15):
				#tmp_dir = './{}/{}/'.format(sample_dir,epoch) 
				#if not os.path.exists(tmp_dir):
				#	os.makedirs(tmp_dir)
								#pdb.set_trace()
			#	save_images([[fake_A[0][0][j]]], [self.batch_size, 1],
			#		'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],j,epoch))



	def test(self, args):
		"""Test cyclegan"""
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		if args.which_direction == 'AtoB':
			sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
		elif args.which_direction == 'BtoA':
			sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
		else:
			raise Exception('--which_direction must be AtoB or BtoA')

		if self.load(args.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		# write html for visual comparison
		index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
		index = open(index_path, "w")
		index.write("<html><body><table><tr>")
		index.write("<th>name</th><th>input</th><th>output</th></tr>")

		out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
			self.testA, self.test_B)

		for sample_file in sample_files:
			print('Processing image: ' + sample_file)
			sample_image = [load_test_data(sample_file)]
			sample_image = np.array(sample_image).astype(np.float32)
			image_path = os.path.join(args.test_dir,
									  '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
			fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
			save_images(fake_img, [1, 1], image_path)
			index.write("<td>%s</td>" % os.path.basename(image_path))
			index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
			'..' + os.path.sep + sample_file)))
			index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
			'..' + os.path.sep + image_path)))
			index.write("</tr>")
		index.close()
