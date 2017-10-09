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
import cv2
import pyflow.pyflow as pyflow
from keras import backend as K
class cyclegan1(object):
	def __init__(self):
		print 1

class cyclegan(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.batch_size = args.batch_size
		self.image_size = args.fine_size
		self.frame_w = args.frame_w 
		self.frame_h = args.frame_h
		self.frames_nb = args.frames_nb
		self.input_c_dim = args.input_nc
		self.input_i_dim = args.input_ic
		self.output_c_dim = args.output_nc
		self.L1_lambda = args.L1_lambda
		self.dataset_dir = args.dataset_dir
		self.frames_nb = args.frames_nb
		self.videogan_generator = videogan_generator 

		self.discriminatorA = discriminator_video
		self.discriminatorB = discriminator_image
		self.z_dim = args.z_dim
				
		if args.use_resnet:
			#self.generatorA = generator_resnet_video
			self.generatorA = videogan_generator
			self.generatorB = None
		else:
			self.generatorA = None
			self.generatorB = None
		if args.use_lsgan:
			#self.criterionGAN = mae_criterion
			self.criterionGAN = abs_criterion
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
										  [None, self.frames_nb, self.frame_h, self.frame_w,
										   self.input_c_dim ],
										  name='real_videos')
		self.real_data_image = tf.placeholder(tf.float32,
											[None, self.image_size, self.image_size,
										   self.input_i_dim],
										  name='real_images')
		
		self.z = tf.placeholder(tf.float32, [None, 100])
		#self.z_st = compress_input2vec(self.real_data_image,self.options,False,"generatorB2A")
		#self.z_om = compress_input2vec(self.real_data_image,self.options,False,"generatorB2A")

		#self.z_st = tf.expand_dims( tf.squeeze(self.z_st) ,0)
		#self.z_om = tf.expand_dims( tf.squeeze(self.z_om) ,0)

		self.fake_A_camera, self.fake_A = self.generatorA(self, self.real_data_image,self.z, self.options, False,name="generatorB2A")
		#self.st_camera = st_camera(self.z_st, self.video_wo_camera, self.options, False, "camera_motion_p")


		self.DA_fake,self.DA_fake_feat = self.discriminatorA(self.fake_A,self.real_data_image,self.options, reuse=False, name="discriminatorA")
		#dump, self.DA_real_feat = self.discriminatorA(self.real_data_video,self.real_data_image,self.options, reuse=True, name="discriminatorA")
		self.real_A = self.real_data_video[:, :, :,:, :]
		self.real_B = self.real_data_image[:, :, :, :]


		self.g_abs_b2a = abs_criterion(self.fake_A, self.real_data_video)
		#self.g_abs_b2a_camera = abs_criterion(self.fake_A_camera, self.real_data_video)
		#self.g_fm_b2a = mse_criterion(self.DA_fake_feat, self.DA_real_feat)
		self.g_dis_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) 
		self.g_loss_b2a = self.g_dis_b2a + self.L1_lambda * self.g_abs_b2a
		

		#<tf.Tensor 'fake_A_sample:0' shape=(1, 15, 112, 112, 3) dtype=float32>      
		self.fake_A_sample = tf.placeholder(tf.float32,
											[None, self.frames_nb, self.frame_w, self.frame_h,
											 self.input_c_dim], name='fake_A_sample')

		#<tf.Tensor 'fake_B_sample:0' shape=(1, 112, 112, 3) dtype=float32>
		self.fake_B_sample = tf.placeholder(tf.float32,
											[None, self.image_size, self.image_size,
											 self.input_i_dim], name='fake_B_sample')



		self.DA_real,_ = self.discriminatorA(self.real_data_video,self.real_data_image, self.options, reuse=True, name="discriminatorA")

		self.DA_fake_sample,_ = self.discriminatorA(self.fake_A_sample,self.real_data_image, self.options, reuse=True, name="discriminatorA")


#		self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
#		self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
#		self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

		self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
		self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
		self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2.0

#		self.g_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
		self.g_abs_b2a_camera_sum = None#tf.summary.scalar("g_abs_b2a_camera_loss",self.g_abs_b2a_camera)
		self.g_b2a_loss_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
		self.g_fm_b2a_sum = None#tf.summary.scalar("g_fm_loss_b2a", self.g_fm_b2a)
		self.g_abs_b2a_sum = tf.summary.scalar("g_abs_b2a_sum", self.g_abs_b2a)
		self.g_dis_b2a_sum = tf.summary.scalar("g_dis_b2a_sum", self.g_dis_b2a)

		self.g_b2a_sum = tf.summary.merge(
			[self.g_b2a_loss_sum,self.g_dis_b2a_sum,self.g_abs_b2a_sum]
		)



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
									 [None, self.frames_nb, self.frame_h, self.frame_w,
									  self.input_c_dim], name='test_A')

		#<tf.Tensor 'test_B:0' shape=(1, 112, 112, 3) dtype=float32>
		self.test_B = tf.placeholder(tf.float32,
									 [None, self.image_size, self.image_size,
									  self.input_i_dim], name='test_B')


		#<tf.Tensor 'generatorA2B_2/Tanh:0' shape=(1, 112, 112, 3) dtype=float32>
#		self.testB = self.generatorB(self.test_A, self.options, True, name="generatorA2B")
		
		#<tf.Tensor 'generatorB2A_2/Tanh:0' shape=(1, 15, 112, 112, 3) dtype=float32>
		#self.testA = self.generatorA(self.test_B, self.options, True, name="generatorB2A")
		self.testA = self.generatorA(self, self.test_B,self.z,  self.options, True,name="generatorB2A")


		t_vars = tf.trainable_variables()
		#self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
		self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
		#self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
		self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
		for var in t_vars: print(var.name)


	def setup_model(self,args):

			self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
				.minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)
					#print 1/0
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2
			self.da_optim = tf.train.AdamOptimizer(args.lr / 10.0, beta1=args.beta1) \
				.minimize(self.da_loss, var_list=self.da_vars)

			#self.db_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			#	.minimize(self.db_loss, var_list=self.db_vars)
			#self.g_a2b_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			#	.minimize(self.g_loss_a2b, var_list=self.g_vars_a2b)

			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2
			#self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
			#	.minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
			init_op = tf.global_variables_initializer()

			self.sess.run(init_op)

			self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
			
			if self.load(args.checkpoint_dir):
				print(" [*] Load SUCCESS")
			else:
				print(" [!] Load failed...") 
						
	def oflow(self,im1,im2):
			alpha = 0.012
			ratio = 0.75
			minWidth = 20
			nOuterFPIterations = 7
			nInnerFPIterations = 1
			nSORIterations = 30
			colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
			u, v, im2W = pyflow.coarse2fine_flow(\
					im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,\
					nSORIterations, colType)
			flow = np.concatenate((u[..., None], v[..., None]), axis=2)
			
			return flow 
		

	def train(self, args):
				self.setup_model(args)         
				counter = 35000
				start_time = time.time()	
									


				"""Train cyclegan"""
				#laurence 
				self.shot_len = 6
				#self.video_bds =\
				#    ['PySceneDetect/488_mp4_shot_boundary.h5.npy',\
				#     'PySceneDetect/489_mp4_shot_boundary.h5.npy',\
				#     'PySceneDetect/490_mp4_shot_boundary.h5.npy',\
				#     'PySceneDetect/499_mp4_shot_boundary.h5.npy',\
				#     'PySceneDetect/498_mp4_shot_boundary.h5.npy']	
				#self.videos = ['PySceneDetect/488.mp4',\
				#               'PySceneDetect/489.mp4',\
				#               'PySceneDetect/490.mp4',\
				#               'PySceneDetect/499.mp4',\
				#               'PySceneDetect/498.mp4']

				self.video_bds =\
					['./slamdunk/001.h5.npy',\
					 './slamdunk/002.h5.npy',\
					 './slamdunk/003.h5.npy',\
					 './slamdunk/004.h5.npy',\
					 './slamdunk/005.h5.npy',\
					 './slamdunk/006.h5.npy',\
					 './slamdunk/007.h5.npy',\
					 './slamdunk/008.h5.npy',\
					 './slamdunk/009.h5.npy',\
					 './slamdunk/010.h5.npy',\
					 './slamdunk/011.h5.npy',\
					 './slamdunk/012.h5.npy',\
					 './slamdunk/013.h5.npy',\
					 './slamdunk/014.h5.npy',\
					 './slamdunk/015.h5.npy',\
					 './slamdunk/016.h5.npy',\
					 './slamdunk/017.h5.npy',\
					 './slamdunk/018.h5.npy',\
					 './slamdunk/019.h5.npy',\
					 './slamdunk/020.h5.npy',\
					 './slamdunk/021.h5.npy',\
					 './slamdunk/022.h5.npy',\
					 './slamdunk/023.h5.npy',\
					 './slamdunk/024.h5.npy',\
					 './slamdunk/025.h5.npy',\
					 './slamdunk/026.h5.npy',\
					 './slamdunk/027.h5.npy',\
					 './slamdunk/028.h5.npy',\
					 './slamdunk/029.h5.npy',\
					 './slamdunk/030.h5.npy']
				self.videos = [\
					 './slamdunk/[52wy][SlamDunk][001][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][002][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][003][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][004][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][005][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][006][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][007][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][008][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][009][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][010][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][011][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][012][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][013][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][014][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][015][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][016][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][017][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][018][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][019][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][020][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][021][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][022][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][023][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][024][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][025][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][026][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][027][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][028][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][029][H264].mp4',\
					 './slamdunk/[52wy][SlamDunk][030][H264].mp4']
		
				epoch = 6         
				for epoch_batch in range(0,200): #args.epoch
					idx = np.random.permutation(len(self.videos))
					list_shot = []
					list_oflow = []
					

					for vid in idx:

						epoch += 1             

						self.bd = np.load(self.video_bds[vid])
						self.count_end = self.bd[-1] - self.shot_len
						cap = cv2.VideoCapture(self.videos[vid])
						start_i = np.random.randint(30,60, size=1)[0]
						merge_image = np.zeros((self.batch_size,self.image_size,self.image_size,self.input_i_dim))

						a_video = np.zeros((self.batch_size,self.frames_nb,self.image_size,self.image_size,self.input_c_dim))		 


						for i in range(start_i,self.count_end - 4,4):
							cap.set(1,i)
							merge_image_one = []								
							a_video_one = []							

							real_image = []
							batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
								.astype(np.float32)
							for j in range(i,i + self.shot_len):
								if j in self.bd:
									a_video_one = []
									break
								ret,frame = cap.read()
								ret,frame = cap.read()
								frame,frame_rs,frames_crop = self.processframe(frame)


								#frame_rs = np.expand_dims(frame_rs,axis=-1)
								if j == i:
									real_image = frame
									m_image = frame_rs
									#ret,frame = cap.read()
									#frame,frame_rs,frames_crop = self.processframe(frame)
								if j == i + self.shot_len - 1 :
									m_image1 = frame_rs
									#ret,frame = cap.read()
									#iframe,frame_rs,frames_crop = self.processframe(frame)
									#m_image = np.concatenate([m_image/127.5 - 1.0, m_image / 127.5 - 1.0],axis=-1)
								if j > i and j < i + self.shot_len - 1:
									a_video_one.append(frames_crop)
								
							
							if a_video_one == []:
								continue

														
							tmp_i = counter % self.batch_size							
							merge_image_one = np.concatenate([m_image / 127.5 - 1 ,m_image1 / 127.5 - 1],axis=-1)
							a_video[tmp_i] = np.array(a_video_one)
							merge_image[tmp_i] = np.array(merge_image_one)
							counter += 1
							if tmp_i != 0:
								continue	


							fake_A = self.sess.run([self.fake_A],feed_dict={self.real_data_image: merge_image,\
									self.z:batch_z})
							# Update G network
							_, summary_str = self.sess.run([self.g_b2a_optim, self.g_b2a_sum],
									feed_dict={self.real_data_image: merge_image,self.real_data_video:a_video,self.z:batch_z})
							self.writer.add_summary(summary_str, counter)
							
							# Update D network
							tmp_rand = np.random.uniform(0,1,1)
							if counter % (2 * self.batch_size):
								_, summary_str = self.sess.run([self.da_optim, self.da_sum],feed_dict={self.real_A: a_video,\
									self.fake_A_sample: fake_A[0],self.real_data_video:a_video,self.real_data_image:merge_image,self.z:batch_z})

								self.writer.add_summary(summary_str, counter)
							
							print(("Epoch: [%2d] [%6d/%6d] time: %4.4f" \
								   % (epoch, i, self.count_end, time.time() - start_time)))
							#if  counter == 2:
							#    self.sample_model(args.sample_dir, epoch, i)

							if np.mod(counter, 50) == 1 or counter ==2:
								#self.validate(epoch,args)
								tmp_dir = './{}/{}/'.format(args.sample_dir,epoch) 
								if not os.path.exists(tmp_dir):
									os.makedirs(tmp_dir)


								save_images([[ np.squeeze(m_image) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,1,epoch))

									#save_images([[ rgb ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,14,epoch))


								make_gif(np.squeeze(np.array(a_video[0])),'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,2,epoch))
								make_gif(np.squeeze(fake_A[0][0]),'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,3,epoch))

								#self.sample_model(args.sample_dir, epoch, i)

							if np.mod(counter, 1000) == 0:
								self.save(args.checkpoint_dir, counter )

						cap.release()


	def processframe(self,frame):
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = np.array(frame[0:650,0:1280])

			frame_rs = cv2.resize(frame, (self.frame_w, self.frame_h)) # / 127.5 -1

			frames_crop = center_crop(frame_rs, 196,196, 256,256) / 127.5 - 1


			noise = np.zeros((256,256,3), np.float32)
			cv2.randn(noise, np.ones(1) * 0, np.ones(1)*0.01)
			frames_crop += noise

			#frames_crop = np.stack((frames_crop,frames_crop,frames_crop),axis=-1)

			return frame, frame_rs, frames_crop 

			frame =  np.expand_dims(frame,-1) 
			frame_rs = np.expand_dims(frame_rs,-1)
			frames_crop = np.expand_dims(frames_crop,-1) 

			frame = np.concatenate([frame,frame,frame],axis=-1)
			frame_rs = np.concatenate([frame_rs,frame_rs,frame_rs],axis=-1)
			frames_crop = np.concatenate([frames_crop,frames_crop,frames_crop],axis=-1)

			
			return frame, frame_rs, frames_crop 


	def validate(self,epoch,args):
			
			start_time = time.time()
			self.video_bds_val =[\
					'./slamdunk/011.h5.npy'\
					]	
			self.videos_val = ['./slamdunk/011.h5.npy'\
					]


			#for epoch in range(200): #args.epoch
			idx = np.random.permutation(len(self.videos_val))


			for vid in idx:

				self.bd = np.load(self.video_bds_val[vid])
				self.count_end_val = self.bd[-1] - self.shot_len
				cap = cv2.VideoCapture(self.videos[vid])
				i = np.random.randint(0,self.count_end_val,1)
				batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
						.astype(np.float32)


				#for i in range(1,self.count_end,4):
				cap.set(1,i)

				a_video = []		 
				m_image = []       
				real_image = []

				for j in range(i,i + self.shot_len):
					if j in self.bd:
						a_video = []
						break
					ret,frame = cap.read()
					frame,frame_rs,frames_crop = self.processframe(frame)

					#frame_rs = np.expand_dims(frame_rs,axis=-1)
					if j == i:
						real_image = frame
						m_image = frame_rs
						ret,frame = cap.read()
						frame,frame_rs,frames_crop = self.processframe(frame)
					if j == i + self.shot_len - 1 :
						m_image1 = frame_rs
						ret,frame = cap.read()
						frame,frame_rs,frames_crop = self.processframe(frame)
						#m_image = np.concatenate([m_image/127.5 - 1.0, m_image / 127.5 - 1.0],axis=-1)
					if j > i and j < i + self.shot_len - 1:
						a_video.append(frames_crop)
								

					a_video.append(frames_crop)
					
					
				if a_video == []:
					continue

				#rgb = self.get_oflow_image(a_video)
				#merge_image = np.concatenate([m_image / 127.5 - 1 ,m_image1 / 127.5 - 1],axis=-1)
				#merge_image = m_image / 127.5 - 1
				#flow = cv2.calcOpticalFlowFarneback(m_image, m_image1, 0.5, 3, 15, 3, 5, 1.2, 0)
				#rgb = self.get_oflow_image(flow) 
				merge_image = np.concatenate([m_image / 127.5 - 1 ,m_image1 / 127.5 - 1],axis=-1)

				fake_A = self.sess.run([self.fake_A],feed_dict={self.real_data_image: [merge_image],self.z:batch_z})
				
				

				print(("Epoch: [%2d] [%6d/%6d] time: %4.4f" \
					   % (epoch, i, self.count_end, time.time() - start_time)))
				#if  counter == 2:
				#    self.sample_model(args.sample_dir, epoch, i)
				tmp_dir = './{}/{}/'.format(args.sample_dir,epoch) 
				if not os.path.exists(tmp_dir):
					os.makedirs(tmp_dir)


				save_images([[ np.squeeze(m_image) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,13,epoch))

				#save_images([[ rgb ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,14,epoch))


				make_gif(np.squeeze(np.array(a_video)),'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,16,epoch))
				make_gif(np.squeeze(fake_A[0][0]),'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,17,epoch))

					#self.sample_model(args.sample_dir, epoch, i)
			cap.release()


			 


	def get_oflow_image(self, flow):
			hsv = np.zeros((self.frame_h,self.frame_w,3), dtype=np.uint8)
			hsv[:, :, 0] = 255
			hsv[:, :, 1] = 255
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			hsv[..., 0] = ang * 180 / np.pi / 2
			hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
			rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			#counter += 1
			#if counter >= 100:
			#    print 1/0
			#    break
			#else:
			#    
			#    tmp_dir = './{}/{}/'.format(args.sample_dir,epoch) 
			#    if not os.path.exists(tmp_dir):
			#        os.makedirs(tmp_dir)
			#    save_images([[ np.squeeze(m_image) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,13,epoch))
			#    save_images([[ rgb ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,14,epoch))
			#    make_gif(np.squeeze(np.array(a_video) * 2 - 1 ),'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,16,epoch))
			#
			return rgb



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

	

		idx = np.random.randint(0,self.count_end,3)
		cap = cv2.VideoCapture('PySceneDetect/488.mp4')

		for i in idx:
			cap.set(1,i)
			

			a_video = []		 
			m_image = []       
			real_image = []
			for j in range(i,i+self.shot_len):
				if j in self.bd:
					a_video = []
					break
				ret,frame = cap.read()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				frame = frame[0:650,0:1280]
				frame_rs = cv2.resize(frame, (128, 128)) / 127.5 -1
				if j == i:
					real_image = frame
					m_image = frame_rs
				a_video.append(frame_rs)

			if a_video == []:
				continue

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


			fake_A = self.sess.run([self.fake_A],feed_dict={self.real_B: [m_image]})
			#fake_A[0][0][0] = video_fh
			#fake_A[0][0][-1] = video_ft


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

			save_images([[ m_image ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,i,13,epoch))
			# save_images([[ video_fh ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],14,epoch))
			# save_images([[ video_ft ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],15,epoch))
			make_gif(a_video,'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(sample_dir,epoch,i,16,epoch))
			make_gif(fake_A[0][0],'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(sample_dir,epoch,i,17,epoch))


			#for j in range(15):
				#tmp_dir = './{}/{}/'.format(sample_dir,epoch) 
				#if not os.path.exists(tmp_dir):
				#	os.makedirs(tmp_dir)
								#pdb.set_trace()
			#	save_images([[fake_A[0][0][j]]], [self.batch_size, 1],
			#		'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(sample_dir,epoch,A[0][:5],j,epoch))
		cap.release()



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
