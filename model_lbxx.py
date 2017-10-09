from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from module import *
from utils import *
import constants as c
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import misc
from sklearn.metrics import mean_squared_error
import cv2
import keras
from keras import backend as K
#from msssim import tf_ms_ssim,tf_ssim
import random
from transformer.spatial_transformer import transformer
from transformer.tf_utils import weight_variable, bias_variable, dense_to_one_hot
import imageio
#b1
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
		#self.videogan_generator = videogan_generator 

		self.discriminatorA = VideoCritic
		self.discriminatorB = discriminator_image
		self.z_dim = args.z_dim
		self.clamp_lower = -0.01
		self.clamp_upper = 0.01

		if args.use_resnet:
			#self.generatorA = generator_resnet_video
			self.generatorA = videogan_generator #videogan_generator_shiftpixel#videogan_generator
			self.generatorB = None
			self.generator_Camera = camera_movement_generator
		else:
			self.generatorA = None
			self.generatorB = None
		if args.use_lsgan:
			#self.criterionGAN = mae_criterion
			self.criterionGAN = mse_criterion#abs_criterion##
		else:
			self.criterionGAN = sce_criterion


		OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
							  gf_dim df_dim output_c_dim frames_nb crop_size')
		self.options = OPTIONS._make((args.batch_size, args.fine_size,
									  args.ngf, args.ndf, args.output_nc, args.frames_nb, args.frame_cs))

		

	def _build_model(self):
		self.real_data_video = tf.placeholder(tf.float32, 
										  [None, self.frames_nb, self.frame_h, self.frame_w, #
										   self.input_c_dim ],
										  name='real_videos')
		self.real_data_image_c = tf.placeholder(tf.float32,
											[None, self.image_size, self.image_size,
										   self.input_i_dim],
										  name='real_c_images')

		self.real_data_image = tf.placeholder(tf.float32,
											[self.options.batch_size, self.image_size, self.image_size,
										   	3],
										  name='real_images')
		self.fake_data_image = tf.placeholder(tf.float32,
											[self.options.batch_size, self.image_size, self.image_size,
										   	3],
										  name='fake_images')
		noise = tf.random_normal(shape = self.real_data_image.get_shape(), mean = 0.0, stddev = 0.005, dtype = tf.float32) 
		self.real_data_image = self.real_data_image + noise 
		self.fake_data_image = self.fake_data_image + noise 
        
		self.z = tf.placeholder(tf.float32, [None, self.z_dim])
		self.diff = tf.placeholder(tf.float32,[None,self.image_size, self.image_size,
										   self.input_i_dim])

		crop_s = 128
		out_size = (crop_s, crop_s)
		mtx0 = tf.tile(tf.constant(np.array([[0.76,0,0,0,0.76,0]])), [self.options.batch_size,1] )
		mtx1 = tf.tile(tf.constant(np.array([[1,0,0,0,1,0]])), [self.options.batch_size,1] )
		self.real_image_crop = tf.reshape(transformer(self.real_data_image,mtx1,(self.options.image_size,self.options.image_size)),[self.options.batch_size,128,128,3])        
		self.real_iamge_merge = tf.concat([self.real_image_crop, self.real_data_image],axis=-1)
        
		self.g4, self.mask1, self.mask2, self.mask3, self.gb, self.fake_A_static, self.m1_gb, self.m2_gf, self.m3_im = self.generatorA(self, self.real_image_crop,self.z, None, self.options, False,name="generatorB2A")
        
		#self.mtx, self.fake_camera_movement = self.generator_Camera(self, self.real_data_image, self.z,\
		#	self.options, False, name="generator_camera")

		self.combined_v = []
		#for i in range(self.frames_nb):
		#	self.combined_v.append( transformer(self.fake_A_static[:,i],mtx1,out_size) )
		#	self.combined_v[i] = tf.expand_dims(self.combined_v[i],1)
            
		self.combined_v = self.fake_A_static
        
		self.combined_v_tf = tf.reshape(tf.concat(self.combined_v,axis=1),[self.batch_size,self.frames_nb,crop_s,crop_s,3])
		self.fake_A = self.combined_v_tf

		self.real_video_tf = self.real_data_video # self.real_video_camera_tf

		self.real_A = self.real_data_video

		self.disc_fake = []
		self.DA_fake = []

        
		self.disc_real, self.DA_real = self.discriminatorA(self.real_video_tf, self.real_image_crop, self.options, reuse=False, name="discriminatorA")         
		self.disc_wi_rv, self.DA_wi_rv = self.discriminatorA(self.real_video_tf, self.fake_data_image, self.options, reuse=True, name="discriminatorA")
		self.disc_fake, self.DA_fake = self.discriminatorA(self.combined_v_tf, self.real_data_image, self.options, reuse=True, name="discriminatorA")
		fake_logit = self.DA_fake
		true_logit = self.DA_real       
		self.c_loss = tf.reduce_mean(fake_logit - true_logit)
		self.gc_loss = tf.reduce_mean(-fake_logit)

        
		self.d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.disc_real , labels=tf.ones_like(self.disc_real) ))
		self.d_loss_wi_rv= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.disc_wi_rv , labels=tf.zeros_like(self.disc_wi_rv) ))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.disc_fake , labels=tf.zeros_like(self.disc_fake) )) 

		self.d_loss = self.d_loss_true + self.d_loss_fake #+ self.d_loss_wi_rv self.c_loss
		self.g_loss_l1 = 10 * tf.reduce_mean(tf.abs(self.real_video_tf - self.combined_v_tf))
		self.g_loss = self.g_loss_l1 + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.disc_fake , labels=tf.ones_like(self.disc_fake) )) # + self.g_loss_l1 self.gc_loss + 

## 
        
## (1,2,/16,0) -- no motion
## (0.01,2,/16,0) -- no motion
## (5.0,2,/16,0)-- no motion
## (5.0,2,/16,2)-- some motion
## (2.0,2,/16,2)-- no motion
## (2.0, 1.5,/16, 1.0) -- no motion
## (0.5, 1.5, /8, 0.1) -- current
## (1, 0.005, 1.0, /16.0, 0.01)



		#self.d_c_loss = tf.reduce_mean(true_c_logit - fake_c_logit)
		#self.g_c_loss = tf.reduce_mean(fake_c_logit)

		###TensorBoard visualization###
		self.z_sum = tf.summary.histogram("z", self.z)
		self.true_sum = tf.summary.histogram("d", true_logit)
		self.fake_sum = tf.summary.histogram("d_", fake_logit)
		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		self.imaginary_sum = video_summary("imaginary", self.fake_A,self.frames_nb)
		#self.d_gp_loss = tf.summary.scalar("d_gp_loss",gradient_penalty)


		self.g_sum = tf.summary.merge([self.z_sum, self.fake_sum, self.imaginary_sum, self.g_loss_sum])
		self.d_sum = tf.summary.merge([self.z_sum, self.true_sum, self.d_loss_sum])


		t_vars = tf.trainable_variables()
		#self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
		self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
		self.g_vars_b2a_camera = [var for var in t_vars if 'generator_camera' in var.name]
		self.d_vars_camera = [var for var in t_vars if 'discriminator_Camera' in var.name]

		#self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
		self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
		self.d_clamp_op = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in self.da_vars]
		self.d_c_clamp_op = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in self.d_vars_camera]

		for var in t_vars: print(var.name)


	def setup_model(self,args):
		self._build_model()		
		self.saver = tf.train.Saver()

		self.pool = ImagePool(args.max_size)
		flag_optim = True
		if flag_optim == True:
			self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
				.minimize(self.g_loss, var_list=self.g_vars_b2a)
			self.g_l1_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
				.minimize(self.g_loss_l1, var_list=self.g_vars_b2a)          
			self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
				.minimize(self.d_loss, var_list=self.da_vars)
			self.d_optim = self.da_optim
			self.g_optim = self.g_b2a_optim

		

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
		counter = 0
		start_time = time.time()	
		flag_camera = 0
							
		"""Train cyclegan"""
		#laurence 
		self.shot_len = 4

		epoch = 22201
		#for epoch_batch in range(0,20000): #args.epoch
		for epoch_batch in range(0,1): #args.epoch
			db = np.load('./slamdunk/lbxx.npz.npy')[0:2000]
			#db = np.load('../slamdunk/lbxx.npz.npy')[2000:]
			pdb.set_trace()
			idx = np.random.permutation(db.shape[0] - 1)
			self.count_end = db.shape[0]
            
			epoch += 1
			i = 0
			for ii in idx:
				i += 1
                
				merge_image = np.zeros((self.batch_size,self.image_size,self.image_size,3))
				merge_image_false = np.zeros((self.batch_size,self.image_size,self.image_size,3))
				a_video = np.zeros((self.batch_size,self.frames_nb,self.image_size,self.image_size,self.input_c_dim))		


				shot = np.array(db[ii]) / 127.5 - 1
				current_frame = shot[0]
				pre_frame = db[ii+1,0]

				if flag_camera == 1:
					merge_image_one = np.concatenate([current_frame, pre_frame, next_frame],axis=-1)
				else:
					merge_image_one = current_frame
					merge_image_one_false = pre_frame

				a_video_one = shot
				first_frame_crop=[]                   

				counter += 1
				tmp_i = counter % self.batch_size
				a_video[tmp_i] = np.array(a_video_one)
					
				merge_image[tmp_i] = np.array(merge_image_one)
				merge_image_false[tmp_i] = np.array(merge_image_one_false)

				if tmp_i != 0:
					continue

				if counter < 10 or counter%500 == 0:
					Diter = 10
				else:
					Diter = 5
						
				pred_video = np.array(a_video)

				batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
					.astype(np.float32)

				if flag_camera == 0:#self.fake_A fake_A
					fake_A, real_video_tf,real_image_tf,real_image_crop,m1_gb,m2_gf,m3_im,mask1,mask2,mask3,g4,gb = self.sess.run([self.fake_A, self.real_video_tf, self.real_data_image, self.real_image_crop,self.m1_gb, self.m2_gf, self.m3_im, self.mask1, self.mask2, self.mask3,self.g4,self.gb ],feed_dict={self.real_data_image: merge_image,self.z:batch_z,self.real_data_video:a_video})
					fake_A = np.array([fake_A])
					pred_video = fake_A[0]                      
					# Update Critic network # #self.d_clamp_op
					######################
					print("====Update Critic====")
					for j in range(1):
						1#_, summary_str, errD, errG, errL1 = self.sess.run([self.d_optim, self.d_sum,self.d_loss,self.g_loss,self.g_loss_l1],\
						#					feed_dict={self.z: batch_z,\
						#					self.fake_data_image: merge_image_false,\
						#					self.real_data_image: merge_image,self.real_data_video:a_video}) #
						#self.writer.add_summary(summary_str, counter) 
					#print("e#rrD: [%4.4f] , errG: [%4.4f], errL1: [%4.4f]" % (errD,errG,errL1))
					####################
					# Update G network #
					####################
					for j in range(1):
						print("====Update Generator====")
						#_, summary_str, errD, errG, errL1 = self.sess.run([self.g_optim, self.g_sum, self.d_loss,self.g_loss,self.g_loss_l1],\
						#					feed_dict={ self.z: batch_z,\
						#					self.fake_data_image: merge_image_false,\
						#					self.real_data_image: merge_image,self.real_data_video:a_video})
						#self.writer.add_summary(summary_str, counter)
					#print("errD: [%4.4f] , errG: [%4.4f], errL1: [%4.4f]" % (errD,errG,errL1))

					print(("Epoch: [%2d] [%6d/%6d] [%9d] time: %4.4f" \
					   % (epoch, i, self.count_end, counter, time.time() - start_time)))
					#if  counter == 2:
					#    self.sample_model(args.sample_dir, epoch, i)
				tmp_rand = random.random()
				if tmp_rand <= 10.1: 
					#self.validate(epoch,args)
					tmp_dir = './{}/{}/'.format(args.sample_dir,epoch) 
					if not os.path.exists(tmp_dir):
						os.makedirs(tmp_dir)
					#np.squeeze(merge_image[0,:,:,0:3]
					save_images([[ np.squeeze(real_image_crop[0]) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,1,epoch))
                        
						#save_images([[ np.squeeze(merge_image[0,:,:,6:9]) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,3,epoch))
						#save_images([[ np.squeeze(fake_A[0][0]) ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,2,epoch))

							#save_images([[ rgb ]],[self.batch_size, 1],'./{}/{}/A_{}_{:02d}_{:03d}.jpg'.format(args.sample_dir,epoch,i,14,epoch))

					make_gif(np.squeeze(np.array(real_video_tf[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,2,epoch),2)
					make_gif(np.array(pred_video[0]),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,3,epoch),2)
                    
					make_gif(np.array(m1_gb[0]),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,4,epoch),2)
					make_gif(np.array(m2_gf[0]),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,5,epoch),2)
					make_gif(np.array(m3_im[0]),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,6,epoch),2)                    
                    
					make_gif(np.squeeze(np.array(mask1[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,7,epoch),2)
					make_gif(np.squeeze(np.array(mask2[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,8,epoch),2)
					make_gif(np.squeeze(np.array(mask3[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,9,epoch),2)
                    
					make_gif(np.squeeze(np.array(g4[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,10,epoch),2)
					make_gif(np.squeeze(np.array(gb[0])),\
							'./{}/{}/A_{}_{:02d}_{:03d}.gif'.format(args.sample_dir,epoch,i,11,epoch),2)
  
                    
				if np.mod(counter, 400) == 0:
					self.save(args.checkpoint_dir, counter )


	def get_bd(self,path):
		bd = np.load(path)
		bd = np.delete(bd,0)
		return bd
	
	def processframe(self,frame0):
			#frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB) / 127.5 -1
			frame = frame0 / 127.5 -1
			#frame = np.array(frame[0:650,0:1280])

			frame_rs = cv2.resize(frame, (self.frame_w, self.frame_h)) # 
			#frame_rs = denoise_tv_chambolle(frame_rs, weight=10)

			frames_crop = center_crop(frame_rs, 196,196, 256,256) 


			#noise = np.zeros((256,256,3), np.float32)
			#cv2.randn(noise, np.ones(1) * 0, np.ones(1)*0.002)
			#frames_crop += noise
			#frame_rs = np.stack([frame_rs,frame_rs,frame_rs],axis=-1) / 127.5 - 1
			#frames_crop = np.stack((frames_crop,frames_crop,frames_crop),axis=-1)
			#pdb.set_trace()

			return frame, frame_rs, frames_crop 

			frame =  np.expand_dims(frame,-1) 
			frame_rs = np.expand_dims(frame_rs,-1)
			frames_crop = np.expand_dims(frames_crop,-1) 

			#frame = np.concatenate([frame,frame,frame],axis=-1)
			#frame_rs = np.concatenate([frame_rs,frame_rs,frame_rs],axis=-1)
			#frames_crop = np.concatenate([frames_crop,frames_crop,frames_crop],axis=-1)

			
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
					if j == i+1:
						m_image2 = frame_rs
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
