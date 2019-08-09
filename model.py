import os
import time

import tensorflow as tf
from glob import glob

from tensorflow.python.data.experimental import shuffle_and_repeat, map_and_batch, prefetch_to_device

from ops import conv, instance_norm, relu, resblock, global_avg_pooling, fully_connected_with_w, global_max_pooling, \
    fully_connected, adaptive_ins_layer_resblock, up_sample, layer_instance_norm, tanh, lrelu, flatten, generator_loss, \
    discriminator_loss, l1_loss, cam_loss, regularization_loss

from utils import check_folder, ImageData


class UGATIT(object):
    def __init__(self, sess, args):

        if args.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.sess = sess
        self.phase = args.phase
        self.dataset_name = args.dataset
        self.light = args.light

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.init_lr = args.lr
        self.GP_ld = args.GP_ld
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.gan_type = args.gan_type

        self.smoothing = args.smoothing

        self.ch = args.ch
        self.n_res = args.n_res  # For the generator
        self.n_dis = args.n_dis  # For the discriminator
        self.n_critic = args.n_critic  # For the discriminator
        self.sn = args.sn  # For the discriminator

        self.image_size = args.image_size
        self.image_ch = args.image_ch
        self.augment_flag = args.augment_flag

        self.chekpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# GAN Type : ", self.gan_type)
        print("# Dataset : ", self.dataset_name)
        print("# Max dataset number : ", self.dataset_num)
        print("# Batch Size : ", self.batch_size)
        print("# Epoch : ", self.epoch)
        print("# Iterations per epoch : ", self.iteration)
        print("# Smoothing : ", self.smoothing)

        print()

        print("##### Generator #####")
        print("# Residual Blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# Discriminator Layer : ", self.n_dis)
        print("# The number of critic layers: ", self.n_critic)
        print("# Spectral Normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# Adversarial Weight : ", self.adv_weight)
        print("# Cycle Weight : ", self.cycle_weight)
        print("# Identity Weight : ", self.identity_weight)
        print("# CAM Weight : ", self.cam_weight)

    def build_model(self):
        if self.phase == 'train':
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            image = ImageData(self.image_size, self.image_ch, self.augment_flag)

            train_a = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            train_b = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

            gpu_device = '/gpu:0'
            train_a = train_a.apply(shuffle_and_repeat(self.dataset_num)) \
                .apply(map_and_batch(image.image_processing, self.batch_size,
                                     num_parallel_batches=16,
                                     drop_remainder=True)) \
                .apply(prefetch_to_device(gpu_device, None))

            train_b = train_b.apply(shuffle_and_repeat(self.dataset_num)) \
                .apply(map_and_batch(image.image_processing,
                                     self.batch_size,
                                     num_parallel_batches=16,
                                     drop_remainder=True)) \
                .apply(prefetch_to_device(gpu_device, None))

            train_a_iterator = train_a.make_one_shot_iterator()
            train_b_iterator = train_b.make_one_shot_iterator()

            self.domain_a = train_a_iterator.next()
            self.domain_b = train_b_iterator.next()

            x_ab, cam_ab = self.generate_a2b(self.domain_a)
            x_ba, cam_ba = self.generate_b2a(self.domain_b)
            
            x_aba, _ = self.generate_b2a(x_ab, reuse=True)
            x_bab, _ = self.generate_a2b(x_ba, reuse=True)
            
            x_aa, cam_aa = self.generate_b2a(self.domain_a, reuse=True)
            x_bb, cam_bb = self.generate_a2b(self.domain_b, reuse=True)
            
            real_a_logit, real_a_cam_logit, real_b_logit, real_b_cam_logit = self.discriminate_real(self.domain_a,
                                                                                                    self.domain_b)
            fake_a_logit, fake_a_cam_logit, fake_b_logit, fake_b_cam_logit  = self.discriminate_fake(x_ba, x_ab)


            if self.gan_type.__contains__('wgan') or self.gan_type == 'drgan':
                gp_a, gp_cam_a = self.gradient_penalty(real=self.domain_a, fake=x_ba, scope='discriminator_A')
                gp_b, gp_cam_b = self.gradient_penalty(real=self.domain_b, fake=x_ab, scope='discriminator_B')
            else:
                gp_a, gp_cam_a = 0, 0
                gp_b, gp_cam_b = 0, 0

            g_ad_loss_a = (generator_loss(self.gan_type, fake_a_logit)
                           + generator_loss(self.gan_type, fake_a_cam_logit))
            g_ad_loss_b = (generator_loss(self.gan_type, fake_b_logit)
                           + generator_loss(self.gan_type, fake_b_cam_logit))

            d_ad_loss_a = (discriminator_loss(self.gan_type, real_a_logit, fake_a_logit)
                           + discriminator_loss(self.gan_type, real_a_cam_logit, fake_a_cam_logit)
                           + gp_a
                           + gp_cam_a)
            d_ad_loss_b = (discriminator_loss(self.gan_type, real_b_logit, fake_b_logit)
                           + discriminator_loss(self.gan_type, real_b_cam_logit, fake_b_cam_logit)
                           + gp_b
                           + gp_cam_b)

            reconstruction_a = l1_loss(x_aba, self.domain_a)
            reconstruction_b = l1_loss(x_bab, self.domain_b)

            identity_a = l1_loss(x_aa, self.domain_a)
            identity_b = l1_loss(x_bb, self.domain_b)

            cam_a = cam_loss(source=cam_ba, non_source=self.domain_a)
            cam_b = cam_loss(source=cam_ab, non_source=self.domain_b)

            generator_a_gan = self.adv_weight * g_ad_loss_a
            generator_a_cycle = self.cycle_weight * reconstruction_a
            generator_a_identity = self.identity_weight * identity_a
            generator_a_cam = self.cam_weight * cam_a

            generator_b_gan = self.adv_weight * g_ad_loss_b
            generator_b_cycle = self.cycle_weight * reconstruction_b
            generator_b_identity = self.identity_weight * identity_b
            generator_b_cam = self.cam_weight * cam_b

            generator_a_loss = generator_a_gan + generator_a_cycle + generator_a_identity + generator_a_cam
            generator_b_loss = generator_b_gan + generator_b_cycle + generator_b_identity + generator_b_cam

            discriminator_a_loss = self.adv_weight * d_ad_loss_a
            discriminator_b_loss = self.adv_weight * d_ad_loss_b

            self.generator_loss = generator_a_loss + generator_b_loss + regularization_loss('generator')
            self.discrimiator_loss = discriminator_a_loss + discriminator_b_loss + regularization_loss('discriminator')

            self.fake_a = x_ba
            self.fake_b = x_ab

            self.real_a = self.domain_a
            self.real_b = self.domain_b

            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if 'generator' in var.name]
            d_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.generator_loss,
                                                                                            var_list=g_vars)
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.discrimiator_loss,
                                                                                            var_list=d_vars)

            self.all_g_losses = tf.summary.scalar('Generator Loss', self.generator_loss)
            self.all_d_losses = tf.summary.scalar('Discriminator Loss', self.discrimiator_loss)

            self.g_a_loss = tf.summary.scalar('G_A_loss', generator_a_loss)
            self.g_a_gan = tf.summary.scalar('G_A_gan', generator_a_gan)
            self.g_a_cycle = tf.summary.scalar('G_A_cycle', generator_a_cycle)
            self.g_a_identity = tf.summary.scalar('G_A_identity', generator_a_identity)
            self.g_a_cam = tf.summary.scalar('G_A_cam', generator_a_cam)

            self.g_b_loss = tf.summary.scalar('G_A_loss', generator_a_loss)
            self.g_b_gan = tf.summary.scalar('G_B_gan', generator_b_gan)
            self.g_b_cycle = tf.summary.scalar('G_B_cycle', generator_b_cycle)
            self.g_b_identity = tf.summary.scalar('G_B_identity', generator_b_identity)
            self.g_b_cam = tf.summary.scalar('G_B_cam', generator_b_cam)

            self.d_a_loss = tf.summary.scalar('d_a_loss', discriminator_a_loss)
            self.d_b_loss = tf.summary.scalar('d_b_loss', discriminator_b_loss)

            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.histogram(var.name + '_min', tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.histogram(var.name + '_max', tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.histogram(var.name + '_mean', tf.reduce_mean(var)))

            g_summary_list = [self.g_a_loss, self.g_a_gan, self.g_a_cam, self.g_a_identity, self.g_a_cycle,
                               self.g_b_loss, self.g_b_gan, self.g_b_cam, self.g_b_identity, self.g_b_cycle,
                               self.all_g_losses]
            g_summary_list.extend(self.rho_var)
            d_summary_list = [self.d_a_loss, self.d_b_loss, self.all_d_losses]

            self.g_loss = tf.summary.merge(g_summary_list)
            self.d_loss = tf.summary.merge(d_summary_list)

        else:
            self.test_domain_a = tf.placeholder(tf.float32, [1, self.image_size, self.image_size, self.image_ch],
                                                name='test_domain_a')
            self.test_domain_b = tf.placeholder(tf.float32, [1, self.image_size, self.image_size, self.image_ch],
                                                name='test_domain_b')

            self.test_fake_a = self.generate_a2b(self.test_domain_a)
            self.test_fake_b = self.generate_b2a(self.test_domain_b)

    def generator(self, x, reuse=False, scope='generator'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channels=channel,
                     pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Downsampling
            for i in range(2):
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)

                channel *= 2

            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_'+str(i))

            # Class Activation Map
            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = relu(x)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            gamma, beta = self.mlp(x, reuse=reuse)

            # Up-sampling
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(x, channel, gamma, beta, smoothing=self.smoothing,
                                                scope='adaptive_resblock'+str(i))

            for i in range(2):
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=4, stride=1, pad=1, pad_type='reflect', scope='up_conv'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm'+str(i))
                x = relu(x)

                channel //= 2

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x, cam_logit, heatmap

    def mlp(self, x, use_bias=True, reuse=False, scope='MLP'):
        channel = self.ch * self.n_res

        if self.light:
            x = global_avg_pooling(x)

        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2):
                x = fully_connected(x, channel, use_bias, scope='linear_'+str(i))
                x = relu(x)

            gamma = fully_connected(x, channel, use_bias=use_bias, scope='gamma')
            beta = fully_connected(x, channel, use_bias=use_bias, scope='beta')

            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, channel])
            beta = tf.reshape(beta, shape=[self.batch_size, 1, 1, channel])

            return gamma, beta

    def generate_a2b(self, x_a, reuse=False):
        out, cam, _ = self.generator(x_a, reuse=reuse, scope='generator_B')
        return out, cam

    def generate_b2a(self, x_b, reuse=False):
        out, cam, _ = self.generator(x_b, reuse=reuse, scope='generator_A')
        return out, cam

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing:
            smoothing = 'smoothing'
        else:
            smoothing = ''

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}'.format(self.model_name, self.dataset_name,
                                                          self.gan_type, n_res, n_dis,
                                                          self.n_critic,
                                                          self.adv_weight, self.cycle_weight,
                                                          self.identity_weight, self.cam_weight,
                                                          sn, smoothing)

    def discriminator(self, x_init, reuse=False, scope='discriminator'):
        d_logit = []
        d_cam_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            local_x, local_cam, local_heatmap = self.discriminator_local(x_init, reuse=reuse, scope='local')
            global_x, global_cam, global_heatmap = self.discriminator_global(x_init, reuse=reuse, scope='global')

            d_logit.extend([local_x, global_x])
            d_cam_logit.extend([local_cam, global_cam])

            return d_logit, d_cam_logit, local_heatmap, global_heatmap

    def discriminator_global(self, x_init, reuse=False, scope='discriminator_global'):
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)
            for i in range(1, self.n_dis-1):
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                        use_bias=True, scope='conv0')
                x = lrelu(x, 0.2)
                channel *= 2

            x = conv(x, channel, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel *= 2

            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, self.sn, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, self.sn, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))
            x = conv(x, channels=1 ,kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap

    def discriminator_local(self, x_init, reuse=False, scope='discriminator_local'):
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='cov_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 2 - 1):
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv'+str(i))
                x = lrelu(x, 0.2)

                channel *= 2

            x = conv(x, channel*2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel *= 2

            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, use_bias=1)
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit') # Classifier

            return x, cam_logit, heatmap

    def discriminate_real(self, x_a, x_b):
        real_a_logit, real_a_cam_logit, _, _ = self.discriminator(x_a, scope='discriminator_a')
        real_b_logit, real_b_cam_logit, _, _  = self.discriminator(x_b, scope='discriminator_b')

        return real_a_logit, real_a_cam_logit, real_b_logit, real_b_cam_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_a_logit, fake_a_cam_logit, _, _ = self.discriminator(x_ba, scope='discriminator_a')
        fake_b_logit, fake_b_cam_logit, _, _ = self.discriminator(x_ab, scope='discriminator_b')

        return fake_a_logit, fake_a_cam_logit, fake_b_logit, fake_b_logit

    def gradient_penalty(self, real, fake, scope='discriminator_a'):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, cam_logit, _, _ = self.discriminator(interpolated, reuse=True, scope=scope)

        gp = []
        cam_gp = []

        for i in range(2):
            grad = tf.gradients(logit[i], interpolated)[0]
            grad_norm = tf.norm(flatten(grad), axis=1)

            if self.gan_type == 'wgan-lp':
                gp.append(self.GP_ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm-1.))))
            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                gp.append(self.GP_ld* tf.reduce_mean(tf.square(grad_norm - 1.)))

        for i in range(2):
            grad = tf.gradients(cam_logit[i], interpolated)[0]
            grad_norm = tf.norm(flatten(grad), axis=1)

            if self.gan_type == 'wgan-lp':
                cam_gp.append(self.GP_ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))
            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                cam_gp.append(self.GP_ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        return sum(gp), sum(cam_gp)

    def train(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        could_load, checkpoint_counter = self.load(self.chekpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter /self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print('[*] Loaded successfully')
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print('[-] Load failed')

        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr

        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag:
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * \
                                                                   (self.epoch - epoch)/(self.epoch - self.decay_epoch)


    def load(self, chekpoint_dir):
        print('[+] Reading checkpoints...')
        checkpoint_dir = os.path.join(chekpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print('[+] Successfully read {}'.format(ckpt_name))

            return True, counter
        else:
            print('[-] Failed to read checkpoint')
            return False, 0
