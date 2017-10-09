from __future__ import division
import tensorflow as tf
from ops import *
#from ops_videogan import *
from utils import *
import ipdb as pdb
import keras
from keras.layers.convolutional import Conv3D
from keras import backend as K
from transformer.spatial_transformer import transformer
from transformer.tf_utils import weight_variable, bias_variable, dense_to_one_hot
	
def discriminator_image(frame, image, options, reuse=False, name="discriminatorA"):

	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		#c3d_1 = Conv3D(128, (3, 3, 3), padding="valid", strides=(1, 1, 1),
		#               activation="relu", name="c3d0")(image)
		

		image1 = tf.concat([image,frame],axis=-1)
		image1 = tf.reshape(image1,[options.batch_size,options.image_size,options.image_size,6])

		h0 = lrelu(conv2d(image1, options.df_dim, name='dd_h0_conv'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='dd_h1_conv'), 'dd_bn1'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='dd_h2_conv'), 'dd_bn2'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*4,  name='dd_h3_conv'), 'dd_bn3'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*4,  name='dd_h4_conv'), 'dd_bn4'))
		h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8,  name='dd_h5_conv'), 'dd_bn5'))
		h6 = lrelu(instance_norm(conv2d(h5, options.df_dim*8,  name='dd_h6_conv'), 'dd_bn6'))
		h7 = lrelu(instance_norm(conv2d(h6, options.df_dim*8,  name='dd_h7_conv'), 'dd_bn7'))
		h8 = lrelu(instance_norm(conv2d(h7, options.df_dim*8,  name='dd_h8_conv'), 'dd_bn8'))

		h9 = linear(tf.contrib.layers.flatten(h8), 8, 'dd_h9_pred')
		h10 = tf.nn.sigmoid(h9)
		return h5,h10



def discriminator_video(video0, options, reuse=False, name="discriminator_Camera"):

	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False


		#image = tf.expand_dims(image,1)
		#image = tf.reshape(image,[tf.shape(video0)[0],2,options.image_size,options.image_size,3])
		#video = tf.concat([video0,image],1)
		video = video0

		h0 = lrelu(conv3d(video, 32,2,5,5,1,2,2, name='dv_h0_conv'))

		h1 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h0, 64,2,5,5,1,2,2, name='dv_h1_conv'), scope='dv_bn1'))
		h2 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h1, 128,2,5,5,1,2,2, name='dv_h2_conv'), scope='dv_bn2'))
		h3 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h2, 128,2,5,5,1,2,2, name='dv_h3_conv'), scope='dv_bn3'))

		h4 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h3, 256,2,5,5,1,2,2, name='dv_h4_conv'), scope='dv_bn4'))
		

		h5 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h4, 256,2,5,5,1,2,2, name='dv_h6_conv'), scope='dv_bn6'))

		h_tmp = tf.contrib.layers.flatten(h5)
		h7 = linear( h_tmp , 10, 'dv_h6_lin')
		#h7 = linear(h7,1,'dv_h7_lin')
		h8 = tf.nn.sigmoid(h7)

		return h8


def VideoCritic( video, image, options, reuse=False, name="VideoCritic"):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()

		i0 = lrelu(conv2d(image, options.df_dim, name='dd_h0_conv'))
		i1 = lrelu(instance_norm(conv2d(i0, options.df_dim*2, name='dd_h1_conv'), 'dd_bn1'))
		i2 = lrelu(instance_norm(conv2d(i1, options.df_dim*4, name='dd_h2_conv'), 'dd_bn2'))
		i3 = lrelu(instance_norm(conv2d(i2, options.df_dim*4,  name='dd_h3_conv'), 'dd_bn3'))
		i4 = lrelu(instance_norm(conv2d(i3, options.df_dim*4,  name='dd_h4_conv'), 'dd_bn4'))
		i5 = tf.contrib.layers.flatten(i4)

		h0 = lrelu(conv3d(video, 64,  name='d_h0_conv'))
		h1 = lrelu(tf.contrib.layers.batch_norm(conv3d(h0, 64*2,  name='d_h1_conv')))
		h2 = lrelu(tf.contrib.layers.batch_norm(conv3d(h1, 64*4,  name='d_h2_conv')))
		h3 = lrelu(tf.contrib.layers.batch_norm(conv3d(h2, 64*8,  name='d_h3_conv')))
		h3 = lrelu(tf.contrib.layers.batch_norm(conv3d(h3, 64*8,  name='d_h4_conv')))
		h4 = tf.concat([i5,tf.contrib.layers.flatten(h3)],axis=-1)

		h4 = linear(h4, 1, 'd_h3_lin')



		return tf.nn.sigmoid(h4), h4


def discriminator_video_old(video0,image, options, reuse=False, name="discriminatorVideo"):

	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False


		#image = tf.expand_dims(image,1)
		#image = tf.reshape(image,[tf.shape(video0)[0],2,options.image_size,options.image_size,3])
		#video = tf.concat([video0,image],1)
		video = video0

		h0 = lrelu(conv3d(video, 32,2,5,5,1,2,2, name='dv_h0_conv'))

		h1 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h0, 64,2,5,5,1,2,2, name='dv_h1_conv'), scope='dv_bn1'))
		h2 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h1, 128,2,5,5,1,2,2, name='dv_h2_conv'), scope='dv_bn2'))
		h3 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h2, 128,2,5,5,1,2,2, name='dv_h3_conv'), scope='dv_bn3'))

		h4 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h3, 256,2,5,5,1,2,2, name='dv_h4_conv'), scope='dv_bn4'))
		

		h5 = lrelu(tf.contrib.layers.batch_norm(
			conv3d(h4, 256,2,5,5,1,2,2, name='dv_h6_conv'), scope='dv_bn6'))
		pdb.set_trace()

		h_tmp = tf.contrib.layers.flatten(h5)
		h7 = linear( h_tmp , 10, 'dv_h6_lin')
		#h7 = linear(h7,1,'dv_h7_lin')
		h8 = tf.nn.sigmoid(h7)

		return h8,h5
def g_background(self, h0, options):
	#with tf.variable_scope(name):
	# image is 256 x 256 x input_c_dim
		#if reuse:
		#	tf.get_variable_scope().reuse_variables()
		#else:
		#	assert tf.get_variable_scope().reuse == False

		#e9 = tf.reshape(h0,(1,1,1,-1))
		e9 = tf.expand_dims(tf.expand_dims(h0,1),1)

		# e8 is (1 x 1 x self.gf_dim*8)
		d1 = deconv2d(tf.nn.relu(e9), options.gf_dim*8, name='g_d1')
		d1 = instance_norm(d1, 'g_bn_d1')
		# d1 is (2 x 2 x self.gf_dim*8*2)

		d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*4, name='g_d2')
		d2 = instance_norm(d2, 'g_bn_d2')
		# d2 is (4 x 4 x self.gf_dim*8*2)

		d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*2, name='g_d3')
		d3 = instance_norm(d3, 'g_bn_d3')
		# d3 is (8 x 8 x self.gf_dim*8*2)

		d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*1, name='g_d4')
		d4 = instance_norm(d4, 'g_bn_d4')
		# d4 is (16 x 16 x self.gf_dim*8*2)

		d5 = deconv2d(tf.nn.relu(d4), 16, name='g_d5')
		d5 = instance_norm(d5, 'g_bn_d5')
		# d5 is (32 x 32 x self.gf_dim*4*2)

		d6 = deconv2d(tf.nn.relu(d5), 8, name='g_d6')
		d6 = instance_norm(d6, 'g_bn_d6')
		# d6 is (64 x 64 x self.gf_dim*2*2)

		d7 = deconv2d(tf.nn.relu(d6), options.output_c_dim, name='g_d7')
		#d7 = instance_norm(d7, 'g_bn_d7')
		# d7 is (128 x 128 x self.gf_dim*1*2)

		#d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
		# d8 is (256 x 256 x output_c_dim)

		return tf.nn.tanh(d7)



def sp_camera(image,z,options,reuse=False,name="sp_camera"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		#h0 = lrelu(conv2d(image, options.df_dim, name='ci_h0_conv_sp'))
		#h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*1,7,7, name='ci_h1_conv_sp'), 'ci_bn1_sp'))
		#h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*2,7,7, name='ci_h2_conv_sp'), 'ci_bn2_sp'))
		#h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*4,6,6, name='ci_h3_conv_sp'), 'ci_bn3_sp'))
		#h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*4,6,6, name='ci_h4_conv_sp'), 'ci_bn4_sp'))
		#h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_sp'), 'ci_bn5_sp'))
	   


		h0 = conv2d(image, options.df_dim,3,2, name='ci_h0_conv_sp')
		h1 = instance_norm(conv2d(lrelu(h0), options.df_dim / 2.0, 3, 2,name='ci_h1_conv_sp'), 'ci_bn1_sp')
		h2 = instance_norm(conv2d(lrelu(h1), options.df_dim / 2.0, 3,2, name='ci_h2_conv_sp'), 'ci_bn2_sp')
		h3 = instance_norm(conv2d(lrelu(h2), options.df_dim,3,2, name='ci_h3_conv_sp'), 'ci_bn3_sp')
		h4 = instance_norm(conv2d(lrelu(h3), options.df_dim,2,2, name='ci_h4_conv_sp'), 'ci_bn4_sp')
		h5 = instance_norm(conv2d(lrelu(h4), options.df_dim*2,2,2, name='ci_h5_conv_sp'), 'ci_bn5_sp')
		h6 = instance_norm(conv2d(lrelu(h5), options.df_dim*2,2,2, name='ci_h6_conv_sp'), 'ci_bn6_sp')
				
		h6_1 = tf.reshape(h6,(tf.shape(h6)[0],options.df_dim*2))
		h6_2 = linear(h6_1,30,'lin0_sp')
		h7 = tf.concat([h6_2,z],axis=-1)

		W1_fc_loc1 = weight_variable([h7.shape[1], 3])
		tmp_s = 0
		tmp_t = 0

		initial1 = np.array([[tmp_s, tmp_t , tmp_t]]).astype('float32').flatten()
		b1_fc_loc1 = tf.Variable(initial_value=initial1, name='b1_fc_loc2_sp')
		feat = tf.nn.tanh(tf.matmul(h7, W1_fc_loc1) + b1_fc_loc1)
		#feat = tf.clip_by_value(feat,0,0.3)

		h1_fc_loc2 = tf.multiply(feat, tf.constant(1.0))

		out_size = (options.crop_size, options.crop_size)
		

		
		tf_s = []
		tf_tx = []
		tf_ty = []
		mtx = []
		h_trans = []

		for i in range(0,8):
			tf_s.append( tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,0],tf.constant(float(i))), tf.constant(0.76),name='sp_s'+str(i)),-1) )
			tf_tx.append( tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,1],tf.constant(float(i) )) , tf.constant(0.0),name='sp_tx'+str(i)),-1) )
			tf_ty.append( tf.expand_dims(tf.add(tf.multiply(h1_fc_loc2[:,2],tf.constant(float(i) )), tf.constant(0.0),name='sp_ty'+str(i)),-1) )
			mtx.append( tf.concat([tf_s[i],tf.zeros([tf.shape(h6)[0],1]),tf_tx[i],tf.zeros([tf.shape(h6)[0],1]),tf_s[i],tf_ty[i]],axis=1) )
			h_trans.append( transformer(image[:,:,:,0:3], mtx[i], out_size) )
			h_trans[i] = tf.reshape( h_trans[i],[tf.shape(h6)[0], 1,options.crop_size,options.crop_size,3] )

		return mtx,h_trans




		



def camera_movement_generator(self, image,z, options, reuse=False, name="generatorC"):
		mtx, v_camera = sp_camera(image,z, options=options,reuse=reuse,name=name)
		return mtx, v_camera


def motion_encoder( diff, options, reuse=False, name='generatorA'):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False
		
		h0 = lrelu(conv2d(diff, options.df_dim, name='ci_h0_conv_st'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='ci_h1_conv_st'), 'ci_bn1_st'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='ci_h2_conv_st'), 'ci_bn2_st'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, name='ci_h3_conv_st'), 'ci_bn3_st'))
		h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*8, name='ci_h4_conv_st'), 'ci_bn4_st'))
		h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*8, name='ci_h5_conv_st'), 'ci_bn5_st'))
		h7 = conv2d(h5, options.df_dim*8, name='cii_h6_pred_st')
		h7 = linear(tf.contrib.layers.flatten(h7), 100, 'ci_h7_lin', with_w=False)
		return h7

def videogan_camera(self,image,z,options,reuse=False,name="generatorC"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False  

def videogan_generator_shiftpixel(self, image,z,mtx, options, reuse = False, name="generatorA"):
	mtx = None
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False 


            
		i0 = lrelu(conv2d(image, 4,4,2, name='ci_i0_conv_om'))
		i1 = lrelu(instance_norm(conv2d(i0, 32 , 4, 2, name='ci_i1_conv_om'), 'ci_bni1_om'))
		i2 = lrelu(instance_norm(conv2d(i1, 64 , 4, 2, name='ci_i2_conv_om'), 'ci_bni2_om'))
		i3 = lrelu(instance_norm(conv2d(i2, 128 , 4, 2,name='ci_i3_conv_om'), 'ci_bni3_om'))
		i4 = lrelu(instance_norm(conv2d(i3, 256 , 4, 2,name='ci_i4_conv_om'), 'ci_bni4_om'))
		i5 = lrelu(instance_norm(conv2d(i4, 512 , 4, 2,name='ci_i5_conv_om'), 'ci_bni5_om'))
		patch_size = 16
		filter_size = 9
		image_patches = tf.extract_image_patches(image, [1, patch_size, patch_size, 1], [1,patch_size,patch_size,1], [1,1,1,1], padding='SAME')
        
		pad = [[0,0],[0,0]]
		patches = tf.space_to_batch_nd(image,[patch_size,patch_size],pad)
		patches = tf.split(patches,patch_size*patch_size,0)
		patches = tf.stack(patches,3)   
		patches = tf.reshape(patches,[self.batch_size,int((self.image_size/patch_size)**2),patch_size,patch_size,3])

		lin1 = tf.contrib.layers.batch_norm(linear(tf.contrib.layers.flatten(i5), (self.frames_nb * (self.image_size / patch_size)**2)*(filter_size**2), 'ci_lin1_om', with_w=False),scope="ci_bnlin1_om" )
        
		predicted_filter = tf.reshape(lin1,[self.batch_size, self.frames_nb, int(self.image_size / patch_size), int(self.image_size / patch_size) , filter_size**2 ])
		predicted_filter = tf.nn.softmax(predicted_filter,dim=-1)
		predicted_filter_flat = tf.reshape(predicted_filter, [self.batch_size, self.frames_nb, -1 ,filter_size,filter_size])
		#image_patches_flat = tf.reshape(image_patches, [self.batch_size, int( (self.image_size/patch_size)**2 ),-1])

		transformed = []
		for i in range(self.batch_size):
			for t in range(self.frames_nb):
				for j in range(int((self.image_size/patch_size)**2) ):
					#image0  = tf.reshape(image_patches_flat[i,j], [1, patch_size, patch_size, 3])
					image0 = tf.expand_dims(patches[i,j],axis=0)
					image0 = tf.pad(image0,[[0,0],[4,4],[4,4],[0,0]],"SYMMETRIC")                   
					#aa = np.zeros([5,5],dtype=np.float32)
					#aa[2,2] = 1                    
					#tmp = tf.constant(predicted_filter_flat[i,t,j]) #predicted_filter_flat[i,t,j]
					tmp = predicted_filter_flat[i,t,j]
					k = tf.reshape(tmp,[filter_size,filter_size,1,1])                    
					k = tf.tile(k,[1,1,3,1])
					t_one = tf.nn.depthwise_conv2d(image0, k, [1, 1, 1, 1], "VALID")
					transformed.append(t_one)
				
		# Using patches here to reconstruct
		transformed0 = tf.concat(transformed, axis=0)
		patches_proc = tf.reshape(transformed0,[-1,int(self.image_size/patch_size),int(self.image_size/patch_size),patch_size**2,3])
		patches_proc = tf.split(patches_proc,patch_size**2,3)
		patches_proc = tf.stack(patches_proc,axis=0)
		patches_proc = tf.reshape(patches_proc,[-1,int(self.image_size/patch_size),int(self.image_size/patch_size),3])
		reconstructed = tf.batch_to_space_nd(patches_proc,[patch_size, patch_size],pad)
		reconstructed = tf.reshape(reconstructed, [self.batch_size , self.frames_nb, self.image_size, self.image_size,3])

		#return gf4, mask1, mask2, mask3, gb4, static_video, m1_gb, m2_gf, m3_im
		return image_patches,transformed0, None, None, None, None, None, reconstructed, None, None, None        
        
        
        
        
def videogan_generator(self, image,z, mtx, options, reuse = False, name="generatorA"):
	mtx = None
	with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False        

		i0 = lrelu(conv2d(image, 4,4,2, name='ci_i0_conv_om'))
		i1 = lrelu(instance_norm(conv2d(i0, 32 , 4, 2, name='ci_i1_conv_om'), 'ci_bni1_om'))
		i2 = lrelu(instance_norm(conv2d(i1, 64 , 4, 2, name='ci_i2_conv_om'), 'ci_bni2_om'))
		i3 = lrelu(instance_norm(conv2d(i2, 128 , 4, 2,name='ci_i3_conv_om'), 'ci_bni3_om'))
		i4 = lrelu(instance_norm(conv2d(i3, 256 , 4, 2,name='ci_i4_conv_om'), 'ci_bni4_om'))
		i5 = lrelu(instance_norm(conv2d(i4, 512 , 4, 2,name='ci_i5_conv_om'), 'ci_bni5_om'))            

		lin1 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(tf.contrib.layers.flatten(i5), 512, 'ci_lin0_om', with_w=False),scope="ci_bnlin0_om" ))
		lin2 = tf.concat([lin1,z],axis = -1)        
        
		i5_ = tf.expand_dims(i5,axis=1)
		i4_ = tf.tile(tf.expand_dims(i4,axis=1),[1,2,1,1,1])
		i3_ = tf.tile(tf.expand_dims(i3,axis=1),[1,4,1,1,1])
		i2_ = tf.tile(tf.expand_dims(i2,axis=1),[1,1,1,1,1])
		i1_ = tf.tile(tf.expand_dims(i1,axis=1),[1,2,1,1,1])
		i0_ = tf.tile(tf.expand_dims(i0,axis=1),[1,4,1,1,1])

		lin1 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(lin1, 16*16*64, 'ci_lin1_om', with_w=False),scope="ci_bnlin1_om"))
		h0 = lin1
        
		h0b = lin1
		de_conv_ks = 5

		d2 = tf.reshape(h0, [-1, 1, 16, 16, 64])
		d1_2 = tf.concat([d2,i2_],axis=-1)                

		d1_3 = deconv3d(d1_2, [self.batch_size, 2, 32, 32, 32],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_b_h1_3', with_w=False)
		d1_3 = tf.nn.relu(tf.contrib.layers.batch_norm(d1_3,scope='g_b_bn1_3_1'))
		d1_3 = tf.concat([d1_3,i1_],axis=-1)                

		d1_4 = deconv3d(d1_3, [self.batch_size, 4, 64, 64, 16],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_b_h1_4', with_w=False)
		d1_4 = tf.nn.relu(tf.contrib.layers.batch_norm(d1_4,scope='g_b_bn1_4_1'))
		d1_4 = tf.concat([d1_4,i0_],axis=-1)              

		h1_5 = deconv3d(d1_4, [self.batch_size, 8, 128, 128, 3],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_b_h1_5', with_w=False)
        
		d7 = h1_5
		gb4 = tf.nn.tanh(d7)

		h2 = tf.reshape(h0, [-1, 1, 16, 16, 64])
		h1_2 = tf.concat([h2,i2_],axis=-1) 

		h1_3 = deconv3d(h1_2, [tf.shape(h0)[0], 2, 32, 32, 32],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_f_h1_3', with_w=False)
		h1_3 = tf.nn.relu(tf.contrib.layers.batch_norm(h1_3,scope='g_f_bn1_3_1'))
		h1_3 = tf.concat([h1_3,i1_],axis=-1)                

		h1_4 = deconv3d(h1_3, [tf.shape(h0)[0], 4, 64, 64, 16],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_f_h1_4', with_w=False)
		h1_4 = tf.nn.relu(tf.contrib.layers.batch_norm(h1_4,scope='g_f_bn1_4_1'))
		h1_4 = tf.concat([h1_4,i0_],axis=-1)                

		h1_5 = deconv3d(h1_4, [tf.shape(h0)[0], 8, 128, 128, 3],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_f_h1_5', with_w=False)

		mask = deconv3d(h1_4,[tf.shape(h0)[0], 8, 128, 128, 3],de_conv_ks,de_conv_ks,de_conv_ks,2,2,2,1,1,1, name='g_mask1', with_w=False)
		mask = tf.nn.softmax(mask,dim=-1)
		mask1 = tf.expand_dims(mask[:,:,:,:,0],axis=-1)
		mask2 = tf.expand_dims(mask[:,:,:,:,1],axis=-1)
		mask3 = tf.expand_dims(mask[:,:,:,:,2],axis=-1)
        
		gf4 = tf.nn.tanh(h1_5)
        
		image_tile = tf.tile(tf.expand_dims(image,axis=1),[1,8,1,1,1])
        
		m1_gb = mask1 * gb4
		m2_gf = mask2 * gf4
		m3_im = mask3 * image_tile
        
        
		static_video = m1_gb  + m2_gf + m3_im

		return gf4, mask1, mask2, mask3, gb4, static_video, m1_gb, m2_gf, m3_im





def g_foreground(self, h0, options):
	#with tf.variable_scope(name):
		# image is 256 x 256 x input_c_dim
		#if reuse:
		#	tf.get_variable_scope().reuse_variables()
		#else:
		#	assert tf.get_variable_scope().reuse == False

		#h0 = tf.reshape(h0,(1,-1))
		h0 = tf.contrib.layers.flatten(h0) 

		l8, self.h0_w, self.h0_b = linear(h0, 256 * 2 * 2 * 1, 'g_f_h0_lin', with_w=True)
		h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, scope='g_f_bn0'))

		h0 = tf.reshape(l8, [-1, 1, 2, 2, 256])


		h1, self.h1_w, self.h1_b = deconv3d(h0, [tf.shape(h0)[0], 2, 4, 4, 256],5,5,5,2,2,2,1,1,1, name='g_f_h1', with_w=True)

		h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, scope='g_f_bn1'))

		h1_1, self.h1_w_1, self.h1_b_1 = deconv3d(h1, [tf.shape(h0)[0], 2, 8, 8, 256],5,5,5,1,2,2,1,1,1, name='g_f_h1_1', with_w=True)
		

		h1_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1_1, scope='g_f_bn1_1'))
		#h1_1 = h1

		h2, self.h2_w, self.h2_b = deconv3d(h1_1, [tf.shape(h0)[0], 2, 16, 16, 128],5,5,5,1,2,2,1,1,1, name='g_f_h2', with_w=True)
		h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, scope='g_f_bn2'))

		h2_1, self.h2_w_1, self.h2_b_1 = deconv3d(h2, [tf.shape(h0)[0], 2, 32, 32, 64],5,5,5,1,2,2,1,1,1, name='g_f_h2_1', with_w=True)
		h2_1 = tf.nn.relu(tf.contrib.layers.batch_norm(h2_1, scope='g_f_bn2_1'))
		#h2_1 = h2


		h3, self.h3_w, self.h3_b = deconv3d(h2_1, [tf.shape(h0)[0], 4, 64, 64, 32],5,5,5,2,2,2,1,1,1, name='g_f_h3', with_w=True)
		h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, scope='g_f_bn3'))

		h3_1, self.h3_w_1, self.h3_b_1 = deconv3d(h3,[tf.shape(h0)[0], 4, 128, 128, 3],5,5,5,1,2,2,1,1,1, name='g_f_h3_1', with_w=True)


		mask1 = deconv3d(h3,[tf.shape(h0)[0], 4, 128, 128, 1],5,5,5,1,2,2,1,1,1, name='g_mask1', with_w=False)
		mask = tf.nn.sigmoid(mask1)

		# mask*h4 + (1 - mask)*

		return tf.nn.tanh(h3_1), mask






def abs_criterion(in_, target):
	return tf.reduce_mean(tf.abs(in_ - target))

def mse_criterion(in_, target):
	return tf.nn.l2_loss(tf.contrib.layers.flatten(in_ - target))
	#return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
