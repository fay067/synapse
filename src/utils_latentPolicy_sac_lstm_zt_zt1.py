from __future__ import division
import tensorflow as tf
import numpy as np
from collections import deque
import random
import tensorflow_probability as tfp
from core import *
from utils import *
from tensorflow.nn.rnn_cell import LSTMStateTuple
import pickle
slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

def test_agent(test_env, sac, MAX_EPISODE_LEN, n=10, render=False):
    returns = []
    
    for j in range(n):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0

        if render: test_env.render()

        while not(d or (ep_len == MAX_EPISODE_LEN)):
            # Take deterministic actions at test time
            o, r, d, _ = test_env.step(sac.get_action(o, True))
            ep_ret += r
            ep_len += 1

            if render: test_env.render()

        returns += [ep_ret]
    
    return np.mean(returns)

EPS = 1e-8
def trun_normal_log_prob(x, mu, std, low, high):
    z = tfd.Normal(0,1).cdf((high-x)/(std+EPS)) - tfd.Normal(0,1).cdf((low-x)/(std+EPS))
    return tf.reduce_sum(-0.5*((x - mu) / (std+EPS))**2 - 0.5*tf.log(2*np.pi) - tf.log(std*z), axis=1, name="log_prob")

class ReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.count = 0
        
    def port_d4rl_data(self, d4rl_data):
        """
        Port d4rl qlearning datasets into buffer
        Now only support running this **before training starts**
        """
        d4rl_size = d4rl_data['rewards'].shape[0]
        if self.max_size < d4rl_size:
            assert False, "Buffer size smaller than the size of d4rl data, cannot port in"
        self.obs1_buf[:d4rl_size] = d4rl_data['observations'].astype(np.float32)
        self.obs2_buf[:d4rl_size] = d4rl_data['next_observations'].astype(np.float32)
        self.acts_buf[:d4rl_size] = d4rl_data['actions'].astype(np.float32)
        self.rews_buf[:d4rl_size] = d4rl_data['rewards'].astype(np.float32)
        self.done_buf[:d4rl_size] = d4rl_data['terminals'].astype(np.float32)
        self.prt = (self.ptr+d4rl_size) % self.max_size
        self.size = min(self.size+d4rl_size, self.max_size)
        self.count += d4rl_size

    def add(self, obs, act, rew, done, next_obs, z=None, z2=None):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.count += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                   )
    
    def save(self, path):
        np.savez(
            path, 
            obs1_buf=self.obs1_buf, 
            obs2_buf=self.obs2_buf, 
            acts_buf=self.acts_buf, 
            rews_buf=self.rews_buf, 
            done_buf=self.done_buf,
        )
        
class ReplayBuffer_Trajectory(object):
# read by traj
    def __init__(self, obs_dim, act_dim, horizon, size):
        # size is in terms of num. of trajectories
        self.obs1_buf = np.zeros([size, horizon, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, horizon, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, horizon, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, horizon], dtype=np.float32)
        self.done_buf = np.zeros([size, horizon], dtype=np.float32)
        self.ptr0, self.ptr1, self.size, self.max_size, self.horizon = 0, 0, 0, size, horizon
        self.count = 0
        
    def port_d4rl_data(self, d4rl_data, obs_mean, obs_std, rew_mean, rew_std):
        """
        Port d4rl sequence datasets (generator format) into buffer
        Now only support running this **before training starts**
        """
        d4rl_data = [_d for _d in d4rl_data] # convert generator to list
        
        d4rl_size = 0
        for i in range(len(d4rl_data)):
#             if d4rl_data[i]['observations'].shape[0] == 1000 and d4rl_data[i]['next_observations'].shape[0]==1000 and  d4rl_data[i]['actions'].shape[0] == 1000:
            d4rl_size += 1
            
        if self.max_size < d4rl_size:
            assert False, "Buffer size smaller than the size of d4rl data, cannot port in"
        
        for i in range(len(d4rl_data)):
#             if d4rl_data[i]['observations'].shape[0] == 1000 and d4rl_data[i]['next_observations'].shape[0] == 1000 and d4rl_data[i]['actions'].shape[0] == 1000:
            self.obs1_buf[self.ptr0, :, :] = (d4rl_data[i]['observations'].astype(np.float32) - obs_mean) / obs_std
            self.obs2_buf[self.ptr0, :, :] = (d4rl_data[i]['next_observations'].astype(np.float32) - obs_mean) / obs_std
            self.acts_buf[self.ptr0, :, :] = d4rl_data[i]['actions'].astype(np.float32)
            self.rews_buf[self.ptr0, :] = (d4rl_data[i]['rewards'].astype(np.float32) - rew_mean) / rew_std
            self.done_buf[self.ptr0, :] = d4rl_data[i]['terminals'].astype(np.float32)
            self.size = min(self.size+1, self.max_size)
            self.ptr0 = (self.ptr0+1) % self.max_size
            self.count += 1

    def add(self, obs, act, rew, done, next_obs):
        self.obs1_buf[self.ptr0, self.ptr1] = obs
        self.obs2_buf[self.ptr0, self.ptr1] = next_obs
        self.acts_buf[self.ptr0, self.ptr1] = act
        self.rews_buf[self.ptr0, self.ptr1] = rew
        self.done_buf[self.ptr0, self.ptr1] = done
        self.ptr1 = (self.ptr1+1) % self.horizon
        if self.ptr1 == 0:
            self.size = min(self.size+1, self.max_size)
            self.ptr0 = (self.ptr0+1) % self.max_size
            self.count += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                   )
    
    def save(self, path):
        np.savez(
            path, 
            obs1_buf=self.obs1_buf, 
            obs2_buf=self.obs2_buf, 
            acts_buf=self.acts_buf, 
            rews_buf=self.rews_buf, 
            done_buf=self.done_buf,
        )
    
"""
Clip gradient whilst handling None error
"""
def ClipIfNotNone(grad, grad_clip_val):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -grad_clip_val, grad_clip_val)

class SAC(object):
    
    def __init__(
        self, 
        graph, 
        sess, 
        state_dim, 
        action_dim, 
        action_bound, 
        save_path, 
        actor_critic=mlp_actor_critic, 
        network_params=dict(), 
        rl_params=dict()
    ):
        self.graph = graph
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.act_limit = action_bound
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.rl_params = rl_params
        self.network_params = network_params
        self.save_path = save_path
        self.replay_size         = rl_params['replay_size']
        self.batch_size          = rl_params['batch_size']
        self.save_freq           = rl_params['save_freq']
        self.render              = rl_params['render']

        # rl params
        self.gamma               = rl_params['gamma']
        self.polyak              = rl_params['polyak']
        self.lr                  = rl_params['lr']
        self.grad_clip_val       = rl_params['grad_clip_val']

        # entropy params
        self.alpha               = rl_params['alpha']
        self.target_entropy      = rl_params['target_entropy']
        
        with self.graph.as_default():
#             # Action limit for clamping: critically, assumes all dimensions share the same bound!
#             act_limit = train_env.action_space.high[0]

            # Share information about action space with policy architecture
            self.network_params['action_scale'] = action_bound[0]

            # Experience buffer
            self.replay_buffer = ReplayBuffer(obs_dim=self.state_dim, act_dim=self.act_dim, size=self.replay_size)
#             self.replay_buffer = buffer

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = placeholders(self.obs_dim, self.act_dim, self.obs_dim, None, None)

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                self.mu, self.pi, self.logp_pi, self.q1_a, self.q2_a  = actor_critic(self.x_ph, self.a_ph, **self.network_params)

            with tf.variable_scope('main', reuse=True):
                # compose q with pi, for pi-learning
                _, _, _, self.q1_pi, self.q2_pi = actor_critic(self.x_ph, self.pi, **self.network_params)

                # get actions and log probs of actions for next states, for Q-learning
                _, self.pi_next, self.logp_pi_next, _, _ = actor_critic(self.x2_ph, self.a_ph, **self.network_params)

            # Target networks
            with tf.variable_scope('target'):
                _, _, _, self.q1_pi_targ, self.q2_pi_targ = actor_critic(self.x2_ph, self.pi_next, **self.network_params)

            # alpha Params
            if self.target_entropy == 'auto':
                self.target_entropy = tf.cast(-self.act_dim, tf.float32)
            else:
                self.target_entropy = tf.cast(self.target_entropy, tf.float32)

            self.log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)

            if self.alpha == 'auto': # auto tune alpha
                self.alpha = tf.exp(self.log_alpha)
            else: # fixed alpha
                self.alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=self.alpha)

            # Count variables
            var_counts = tuple(count_vars(scope) for scope in ['log_alpha',
                                                               'main/pi',
                                                               'main/q1',
                                                               'main/q2',
                                                               'main'])
#             print("""\nNumber of other parameters:
#                      alpha: %d,
#                      pi: %d,
#                      q1: %d,
#                      q2: %d,
#                      total: %d\n"""%var_counts)

            # Min Double-Q:
            self.min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)
            self.min_q_pi_targ = tf.minimum(self.q1_pi_targ, self.q2_pi_targ)

            # Targets for Q and V regression
            self.q_backup = tf.stop_gradient(self.r_ph + self.gamma*(1-self.d_ph)*(self.min_q_pi_targ - self.alpha*self.logp_pi_next))

            # critic losses
            self.q1_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q1_a)**2)
            self.q2_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q2_a)**2)
            self.value_loss = self.q1_loss + self.q2_loss

            # Soft actor losses
            self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.min_q_pi)

            # alpha loss for temperature parameter
            self.alpha_backup = tf.stop_gradient(self.logp_pi + self.target_entropy)
            self.alpha_loss  = -tf.reduce_mean(self.log_alpha * self.alpha_backup)

            # Policy train op
            # (has to be separate from value train op, because q1_logits appears in pi_loss)
            self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-04)
            if self.grad_clip_val is not None:
                gvs = self.pi_optimizer.compute_gradients(self.pi_loss,  var_list=get_vars('main/pi'))
                capped_gvs = [(ClipIfNotNone(grad, self.grad_clip_val), var) for grad, var in gvs]
                self.train_pi_op = self.pi_optimizer.apply_gradients(capped_gvs)
            else:
                self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))

            # Value train op
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-04)
            with tf.control_dependencies([self.train_pi_op]):
                if self.grad_clip_val is not None:
                    gvs = self.value_optimizer.compute_gradients(self.value_loss, var_list=get_vars('main/q'))
                    capped_gvs = [(ClipIfNotNone(grad, self.grad_clip_val), var) for grad, var in gvs]
                    self.train_value_op = self.value_optimizer.apply_gradients(capped_gvs)
                else:
                    self.train_value_op = self.value_optimizer.minimize(self.value_loss, var_list=get_vars('main/q'))

            self.alpha_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-04)
            with tf.control_dependencies([self.train_value_op]):
                self.train_alpha_op = self.alpha_optimizer.minimize(self.alpha_loss, var_list=get_vars('log_alpha'))

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([self.train_value_op]):
                self.target_update = tf.group([tf.assign(v_targ, self.polyak*v_targ + (1-self.polyak)*v_main)
                                          for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            # All ops to call during one training step
            self.step_ops = [self.pi_loss, self.q1_loss, self.q2_loss, self.q1_a, self.q2_a, self.logp_pi, self.target_entropy, self.alpha_loss, self.alpha, self.train_pi_op, self.train_value_op, self.train_alpha_op, self.target_update]

            # Initializing targets to match main variables
            self.target_init = tf.group([tf.assign(v_targ, v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
            
            self.act_op = self.pi
            self.act_op_deterministic = self.mu
            
            self.saver = tf.train.Saver()
            
        self.last_num_epi = -1

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)
        
    def get_action(self, o, deterministic=False):
        if deterministic:
            return self.sess.run(self.act_op_deterministic, feed_dict={self.x_ph: o.reshape(1,-1)})[0]
        return self.sess.run(self.act_op, feed_dict={self.x_ph: o.reshape(1,-1)})[0]
    
    def get_action_eval_batch(self, o, deterministic=False):
        if deterministic:
            return self.sess.run(self.act_op_deterministic, feed_dict={self.x_ph: o.reshape(-1,self.state_dim)})
        return self.sess.run(self.act_op, feed_dict={self.x_ph: o.reshape(-1,self.state_dim)})
    
    def train(self, batch, num_epi):
        if (num_epi+1)%self.save_freq == 0 and num_epi!=self.last_num_epi:
            self.saver.save(self.sess, self.save_path)
            self.last_num_epi = num_epi
            
        feed_dict = {
            self.x_ph: batch['obs1'],
            self.x2_ph: batch['obs2'],
            self.a_ph: batch['acts'],
            self.r_ph: batch['rews'],
            self.d_ph: batch['done'],   
        }
        return self.sess.run(self.step_ops, feed_dict)
        

class OPE_Model(object):
    def __init__(self, graph, sess, lr, ds, dr, code_size, state_dim, state_bound, action_dim, save_appendix, buffer_size, buffer_seed, minibatch_size, horizon, beta=1., is_training=True, lstm_hidden=64):
        self.graph = graph
        self.sess = sess
        self.code_size = code_size
        self.num_hidden = lstm_hidden
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.horizon = horizon
        self.save_appendix = os.path.join("./saved_model/", save_appendix, "ope.ckpt")
        self.replay_buffer = ReplayBuffer_Trajectory(state_dim, action_dim, horizon, buffer_size)
        if state_bound != None:
            self.state_bound = tf.constant(state_bound, dtype=tf.float32, name="state_bound")
        else:
            self.state_bound = None
        self.beta = beta
        
        with self.graph.as_default():
            
            self.state_holder = tf.placeholder(
                shape=[None, state_dim], 
                dtype=tf.float32, 
                name='Encoder_state_holder'
            ) 
            
            self.state2_holder = tf.placeholder(
                shape=[None, state_dim], 
                dtype=tf.float32, 
                name='Encoder_state2_holder'
            ) 
            
            self.action_holder = tf.placeholder(
                shape=[None, action_dim], 
                dtype=tf.float32, 
                name='Encoder_action_holder'
            ) 
            
            self.r_holder = tf.placeholder(
                shape=[None, 1], 
                dtype=tf.float32, 
                name='Encoder_r_holder'
            ) 
            
            self.encoder_lstm_state_holder = tf.placeholder(
                shape=[None, self.num_hidden*2], 
                dtype=tf.float32, 
                name='Encoder_lstm_state_holder'
            ) 
            
            self.decoder_lstm_state_holder = tf.placeholder(
                shape=[None, self.num_hidden*2], 
                dtype=tf.float32, 
                name='Decoder_lstm_state_holder'
            ) 
            
            self.decoder_zt_holder = tf.placeholder(
                shape=[None, code_size], 
                dtype=tf.float32, 
                name='Decoder_zt_holder'
            ) 
            
            self.t0_holder = tf.placeholder(
                shape=[None, 1], 
                dtype=tf.float32, 
                name='t0_holder'
            ) 
            
            # Zt from t=0 to T-2; Zt+1 from t=1 to T-1
            self.encoder_zt_out_from_st_dist, self.encoder_zt_out_from_st_sample, self.encoder_zt1_out_from_st1_dist, self.encoder_zt1_out_from_st1_sample = self.make_encoder_zt_zt1_from_st_st1(self.state_holder, self.state2_holder, is_training=is_training)
            
            print ("self.encoder_zt_out_from_st_dist", self.encoder_zt_out_from_st_dist)
            print ("self.encoder_zt_out_from_st_sample", self.encoder_zt_out_from_st_sample)
            print ("self.encoder_zt1_out_from_st1_dist", self.encoder_zt1_out_from_st1_dist)
            print ("self.encoder_zt1_out_from_st1_sample", self.encoder_zt1_out_from_st1_sample)
            
            
            with tf.control_dependencies([self.encoder_zt_out_from_st_sample, self.encoder_zt1_out_from_st1_sample]):
            
            # t=1 to T-1
                self.encoder_zt1_out_from_zt_dist, self.encoder_zt1_out_from_zt_sample, self.encoder_lstm_state = self.make_encoder_zt1_from_zt(
                    self.encoder_zt_out_from_st_sample,
                    self.state2_holder,
                    self.action_holder,
                    self.encoder_lstm_state_holder,
                    is_training=is_training
                )

                print ("self.encoder_zt1_out_from_zt_dist", self.encoder_zt1_out_from_zt_dist)
                print ("self.encoder_zt1_out_from_zt_sample", self.encoder_zt1_out_from_zt_sample)
                print ("self.encoder_lstm_state", self.encoder_lstm_state)
                

        # Zt+1 from t=1 to T-1
        self.decoder_prior_dist = self.make_prior()
        self.decoder_prior_sample = self.decoder_prior_dist.sample()
#         self.decoder_prior_sample = tf.ones((self.minibatch_size, self.code_size), name="prior_z0", dtype=tf.float32)
        self.decoder_zt1_dist, self.decoder_zt1_sample, self.decoder_lstm_state, self.decoder_zt1_mean = self.make_decoder_zt1(self.decoder_zt_holder, self.action_holder, self.decoder_lstm_state_holder, is_training=is_training)
        
        self.decoder_state_dist, self.decoder_state_sample, self.decoder_state_mean = self.make_decoder_state(self.decoder_zt_holder, is_training=is_training)

        print ("self.decoder_zt1_sample", self.decoder_zt1_sample)
        print ("self.decoder_zt1_dist", self.decoder_zt1_dist)
        print ("self.decoder_lstm_state", self.decoder_lstm_state)

        with tf.control_dependencies([self.decoder_zt1_sample, self.decoder_lstm_state]):
        # t=0 to T-2
            self.decoder_state2_log_prob, self.decoder_state2_sample, self.decoder_state2_mean = self.make_decoder_state(self.decoder_zt1_sample, is_training=is_training)
            self.decoder_r_dist, self.decoder_r_sample, self.decoder_r_mean = self.make_decoder_reward(self.decoder_zt_holder, self.decoder_zt1_sample, self.action_holder, is_training=is_training)

            print ("self.decoder_state_log_prob", self.decoder_state2_log_prob)
            print ("self.decoder_state_sample", self.decoder_state2_sample)
            print ("self.decoder_r_diss", self.decoder_r_dist)
            print ("self.decoder_r_sample", self.decoder_r_sample)
        
        with tf.control_dependencies([self.decoder_state2_sample, self.decoder_r_sample, self.encoder_zt1_out_from_zt_sample, self.encoder_zt1_out_from_st1_sample, self.decoder_zt1_sample, self.encoder_zt_out_from_st_sample, self.decoder_lstm_state, self.encoder_lstm_state]):
            self.likelihood_s = tf.reduce_mean(self.decoder_state2_log_prob)

            self.likelihood_r = tf.reduce_mean(self.decoder_r_dist.log_prob(self.r_holder))

            self.divergence1 = tf.reduce_mean(tfd.kl_divergence(self.encoder_zt1_out_from_zt_dist, self.encoder_zt1_out_from_st1_dist))

            self.divergence2 = tf.reduce_mean(tfd.kl_divergence(self.encoder_zt1_out_from_zt_dist, self.decoder_zt1_dist))
            
            if self.beta < 0.05:
                self.divergence2 = tf.clip_by_value(self.divergence2, 0., 100.)

            self.divergence3 = tf.reduce_mean(tf.multiply(self.t0_holder, tfd.kl_divergence(self.encoder_zt_out_from_st_dist, self.decoder_prior_dist)))

            self.encoder_decoder_lstm_states_mse = tf.reduce_mean(tf.losses.mean_pairwise_squared_error(self.decoder_lstm_state, self.encoder_lstm_state))

            print ("self.likelihood_s", self.likelihood_s)
            print ("self.likelihood_r", self.likelihood_r)
            print ("self.divergence1", self.divergence1)
            print ("self.divergence2", self.divergence2)
#             print "self.divergence3", self.divergence3
            print ("self.encoder_decoder_lstm_states_mse", self.encoder_decoder_lstm_states_mse)

            self.elbo = - self.beta*self.divergence1 - self.beta*self.divergence2 - self.beta*self.divergence3 + self.likelihood_s + self.likelihood_r - self.encoder_decoder_lstm_states_mse
#         self.elbo = - self.beta*self.divergence1 - self.beta*self.divergence2 + self.likelihood_s + self.likelihood_r - self.encoder_decoder_lstm_states_mse
        self.global_step = tf.Variable(0., trainable=False, name="training_step")
        self.global_step_increment = self.global_step.assign(self.global_step+1)
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps=ds, decay_rate=dr)

        self.regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops + [self.elbo]):
            self.optimize = tf.train.AdamOptimizer(
                self.learning_rate
            )

            self.optimize_gradients = self.optimize.compute_gradients(
                -self.elbo
                +tf.reduce_mean(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) 
                )
            )

            self.optimize_clipped_gradients = [
                (tf.clip_by_value(grad, -10., 10.), var)
                if (var.name.find("scale")!=-1)
                else (grad, var)
                for (grad, var) in self.optimize_gradients
            ]

            self.optimizer = self.optimize.apply_gradients(self.optimize_clipped_gradients)
                
        self.saver = tf.train.Saver()

            
    def make_encoder_zt_zt1_from_st_st1(self, state, state2, reuse=tf.AUTO_REUSE, is_training=True, var_scope="Encoder_zt"):
        with tf.variable_scope(var_scope, reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                x = slim.fully_connected(tf.stack([state, state2]), 128, scope="fc1")
                x = slim.fully_connected(x, 64, scope="fc2")
                loc = slim.fully_connected(x, self.code_size, activation_fn=None, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="loc")
                scale = slim.fully_connected(x, self.code_size, activation_fn=tf.nn.softplus, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="scale")
                
                zt_dist = tfd.MultivariateNormalDiag(loc[0], scale[0])
                zt_sample = zt_dist.sample()
                zt1_dist = tfd.MultivariateNormalDiag(loc[1], scale[1])
                zt1_sample = zt1_dist.sample()
                
                return zt_dist, zt_sample, zt1_dist, zt1_sample
        
            
    def make_encoder_zt1_from_zt(self, zt, state2, action, lstm_state, reuse=tf.AUTO_REUSE, is_training=True, var_scope="Encoder_zt1"):
        with tf.variable_scope(var_scope, reuse=reuse) as scope:
            lstm_state_in = LSTMStateTuple(lstm_state[:, :self.num_hidden], lstm_state[:, self.num_hidden:])
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, reuse=reuse)
            outputs, last_lstm_state = tf.nn.dynamic_rnn(lstm_cell, tf.stack([tf.concat([zt, state2, action], 1)]), initial_state=lstm_state_in, dtype=tf.float32, time_major=True)
            lstm_hidden_states = tf.concat(last_lstm_state, 1)
        with tf.variable_scope(var_scope, reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):

                fc1 = slim.fully_connected(outputs[0], 64, scope="fc1")
                loc = slim.fully_connected(fc1, self.code_size, activation_fn=None, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="loc")
                scale = slim.fully_connected(fc1, self.code_size, activation_fn=tf.nn.softplus, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="scale")
                dist = tfd.MultivariateNormalDiag(loc, scale)
                return dist, dist.sample(), lstm_hidden_states
        
    def make_prior(self):
        loc = tf.zeros((self.minibatch_size, self.code_size))
        scale = tf.ones((self.minibatch_size, self.code_size))
        return tfd.MultivariateNormalDiag(loc, scale)
            
    def make_decoder_zt1(self, zt, action, lstm_state, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("Decoder_zt1", reuse=reuse) as scope:
            zt = [zt]
            action = [action]
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, reuse=reuse)
            outputs, new_lstm_state = tf.nn.dynamic_rnn(lstm_cell, tf.stack([tf.concat(zt+action, 1)]), dtype=tf.float32, initial_state=LSTMStateTuple(lstm_state[:,:self.num_hidden], lstm_state[:,self.num_hidden:]), time_major=True)
            new_lstm_state = tf.concat(new_lstm_state, 1)
            print ("new_lstm_state", new_lstm_state)
            
            def map_decoder_lstm_to_encoder(l_in, reuse=reuse, is_training=is_training):
                with tf.variable_scope("map_lstm_states", reuse=reuse) as scope:
                    with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                        fc1 = slim.fully_connected(l_in, 128, scope="fc1")
                        fc2 = slim.fully_connected(fc1, 128, scope="fc2")
                        o = slim.fully_connected(fc2, 2*self.num_hidden, scope="out")
                        return o
                    
            new_lstm_state = map_decoder_lstm_to_encoder(new_lstm_state)
            
            with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                fc1 = slim.fully_connected(outputs[0], 64, scope="fc1")
                loc = slim.fully_connected(fc1, self.code_size, activation_fn=None, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="loc")
                scale = slim.fully_connected(fc1, self.code_size, activation_fn=tf.nn.softplus, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="scale")
                out_dist = tfd.MultivariateNormalDiag(loc, scale)
                out_sample = out_dist.sample()
                return out_dist, out_sample, new_lstm_state, loc
            
    def make_decoder_state(self, zt, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("Decoder_state", reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                fc1 = slim.fully_connected(zt, 128, scope="fc1")
                fc2 = slim.fully_connected(fc1, 64, scope="fc2")
                loc_state = slim.fully_connected(fc2, self.state_dim, activation_fn=None, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="loc_state")
                scale_state = slim.fully_connected(fc2, self.state_dim, activation_fn=tf.nn.softplus, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="scale_state")
                # We could sample from tfd.TruncatedNormal, however, it could cause gradients to become NaN sometimes.
                # So the log_probs have to be implemented from scratch.
                self.loc_state = loc_state
                self.scale_state = scale_state
                if self.state_bound == None:
                    out_dist = tfd.MultivariateNormalDiag(loc_state, scale_state)
                    out_sample = out_dist.sample()
                    out_log_prob = out_dist.log_prob(self.state2_holder)
                else:
                    out_sample = tfd.TruncatedNormal(loc_state, scale_state, -self.state_bound, self.state_bound).sample()
                    out_log_prob = trun_normal_log_prob(self.state2_holder, loc_state, scale_state, -self.state_bound, self.state_bound)
                return out_log_prob, out_sample, loc_state
            
                
    def make_decoder_reward(self, zt, zt1, action, reuse=tf.AUTO_REUSE, is_training=True):
            
        with tf.variable_scope("Decoder_reward", reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected], 
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.glorot_uniform_initializer,
                                    weights_regularizer=slim.l2_regularizer(0.001),
                                    biases_regularizer=slim.l2_regularizer(0.001),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                fc1_zt = slim.fully_connected(zt, 128, scope="fc1_zt")
                fc1_a = slim.fully_connected(action, 128, scope="fc1_action")
                fc1_zt1 = slim.fully_connected(zt1, 128, scope="fc1_zt1")
                fc2 = slim.fully_connected(tf.concat([fc1_zt, fc1_zt1, fc1_a], axis=1), 64, scope="fc2")
                loc_r = slim.fully_connected(fc2, 1, activation_fn=None, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="loc_reward")
                scale_r = slim.fully_connected(fc2, 1, activation_fn=tf.nn.softplus, normalizer_fn=None, weights_regularizer=None, biases_regularizer=None, biases_initializer=None, scope="scale_reward")
                self.loc_r = loc_r
                self.scale_r = scale_r
                out_dist = tfd.MultivariateNormalDiag(loc_r, scale_r)
                out_sample = out_dist.sample()
                return out_dist, out_sample, loc_r
            
    def init_z0_s0(self):
        self.zt = self.sess.run(self.decoder_prior_sample)
        self.zt = self.zt[0].reshape(1,-1)
        s0 = self.sess.run(self.decoder_state_sample, feed_dict={self.decoder_zt_holder : self.zt})
        self.encoder_lstm = np.zeros((1, self.num_hidden*2)).astype(np.float32)
        self.decoder_lstm = np.zeros((1, self.num_hidden*2)).astype(np.float32)
        return s0[0]
    
    def get_zt1_s2_r(self, action):
        self.zt1, self.decoder_lstm, s2, r = self.sess.run(
            [self.decoder_zt1_mean, self.decoder_lstm_state, self.decoder_state2_mean, self.decoder_r_mean],
            feed_dict = {
                self.decoder_lstm_state_holder : self.decoder_lstm,
                self.action_holder : action,
                self.decoder_zt_holder : self.zt,
            }
        )
        return s2[0], r[0][0]
    
    def get_zt1_s2_r_debug(self, action):
        self.zt1, self.decoder_lstm, s2, r, loc_state, scale_state, loc_r, scale_r = self.sess.run(
            [self.decoder_zt1_mean, self.decoder_lstm_state, self.decoder_state2_mean, self.decoder_r_mean, self.loc_state, self.scale_state, self.loc_r, self.scale_r],
            feed_dict = {
                self.decoder_lstm_state_holder : self.decoder_lstm,
                self.action_holder : action,
                self.decoder_zt_holder : self.zt,
            }
        )
        return s2[0], r[0][0], loc_state, scale_state, loc_r, scale_r
    
    def update_zt(self):
        self.zt = np.copy(self.zt1)
            
        
    def train(self, batch):
        
        (
            s_batch, a_batch, r_batch, t_batch, 
            s2_batch
        ) = (
            batch["obs1"],
            batch["acts"],
            batch["rews"],
            batch["done"],
            batch["obs2"],
        )
        
        for _t in range(self.horizon):
            if _t == 0:
                zt = self.sess.run(self.decoder_prior_sample)
#                 print "z0", zt
                encoder_lstm = np.zeros((self.minibatch_size, self.num_hidden*2)).astype(np.float32)
#                 print "encoder_lstm", encoder_lstm
                decoder_lstm = np.zeros((self.minibatch_size, self.num_hidden*2)).astype(np.float32)
#                 print "decoder_lstm", decoder_lstm
                t0s = np.ones((self.minibatch_size, 1))
                self.elbo_evaluated = []
                self.likelihood_s_evaluated = []
                self.likelihood_r_evaluated = []
                self.divergence1_evaluated = []
                self.divergence2_evaluated = []
                self.divergence3_evaluated = []
            else:
                t0s = np.zeros((self.minibatch_size, 1))
                
            
            encoder_lstm, decoder_lstm, zt1, likelihood_s_evaluated, likelihood_r_evaluated, divergence1_evaluated, divergence2_evaluated, divergence3_evaluated, self.encoder_decoder_lstm_states_mse_evaluated, elbo_evaluated, clipped_grads, _, _ = self.sess.run(
                [self.encoder_lstm_state, self.decoder_lstm_state, self.decoder_zt1_sample, self.likelihood_s, self.likelihood_r, self.divergence1, self.divergence2, self.divergence3, self.encoder_decoder_lstm_states_mse, self.elbo, self.optimize_clipped_gradients, self.optimizer, self.global_step_increment], 
                feed_dict={
                    self.action_holder : a_batch[:, _t, :], 
                    self.state2_holder : s2_batch[:, _t, :],
                    self.r_holder : r_batch[:, _t].reshape(self.minibatch_size, 1),
                    self.state_holder : s_batch[:, _t, :],
                    self.t0_holder: t0s,
                    self.encoder_lstm_state_holder : encoder_lstm,
                    self.decoder_lstm_state_holder : decoder_lstm,
                    self.decoder_zt_holder : zt,
                }
             )

            self.elbo_evaluated += [elbo_evaluated]
            self.likelihood_s_evaluated += [likelihood_s_evaluated]
            self.likelihood_r_evaluated += [likelihood_r_evaluated]
            self.divergence1_evaluated += [divergence1_evaluated]
            self.divergence2_evaluated += [divergence2_evaluated]
            self.divergence3_evaluated += [divergence3_evaluated]
            zt = zt1
#         print "zt", zt
#         print "encoder_lstm", encoder_lstm
#         print "decoder_lstm", decoder_lstm

            if (self.sess.run(self.global_step) + 1) % 100 == 0:
                self.saver.save(self.sess, self.save_appendix)



        
            

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
class D4RL_Policy:
    """D4RL policy for evaluation."""

    def __init__(self, policy_file):
        with tf.io.gfile.GFile(os.path.join("gs://gresearch/deep-ope/d4rl", policy_file), 'rb') as f:
            weights = pickle.load(f)
#         weights = np.load(policy_file)
        self.fc0_w = weights['fc0/weight']
        self.fc0_b = weights['fc0/bias']
        self.fc1_w = weights['fc1/weight']
        self.fc1_b = weights['fc1/bias']
        self.fclast_w = weights['last_fc/weight']
        self.fclast_b = weights['last_fc/bias']
        self.fclast_w_logstd = weights['last_fc_log_std/weight']
        self.fclast_b_logstd = weights['last_fc_log_std/bias']
        relu = lambda x: np.maximum(x, 0)
        self.nonlinearity = np.tanh if weights['nonlinearity'] == 'tanh' else relu

        identity = lambda x: x
        self.output_transformation = np.tanh if weights['output_distribution'] == 'tanh_gaussian' else identity

    def act(self, state, noise=0.):
        x = np.dot(self.fc0_w, state) + self.fc0_b
        x = self.nonlinearity(x)
        x = np.dot(self.fc1_w, x) + self.fc1_b
        x = self.nonlinearity(x)
        mean = np.dot(self.fclast_w, x) + self.fclast_b
        logstd = np.dot(self.fclast_w_logstd, x) + self.fclast_b_logstd

        action = self.output_transformation(mean + np.exp(logstd) * noise)
        return action, mean



