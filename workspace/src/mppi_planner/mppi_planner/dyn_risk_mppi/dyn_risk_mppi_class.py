#!/usr/bin/env python3
import os
# from typing import Tuple
# Jax utility functions
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # 0.9 causes too much lag. 
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import jax
import jax.numpy as jnp
import numpy as np
from jax import config  
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', 'high')
import functools


# https://arxiv.org/pdf/2307.09105 
# Utility function needed for Halton Splines instead for better exploration and smoother trajectories
import ghalton

from mppi_planner.models import DifferentialDrive
from mppi_planner.smoother import TrajectorySmoother
from mppi_planner.critics import GoalCritic, GoalAngleCritic, DynObstaclesCritic, PreferForwardCritic, PathAngleCritic
class MPPI:
    def __init__(self,cfg, mppi_key):
        """
        cfg: full loaded YAML config (dict)

        """
        #load the configs
        self.cfg = cfg  

        # -----------------------
        # MPPI core config
        # -----------------------
        mppi_cfg = cfg["mppi"]

        self.dim_st = int(mppi_cfg["dim_state"])
        self.dim_ctrl = int(mppi_cfg["dim_control"])
        self.dim_euclid = int(mppi_cfg["dim_euclid"])

        self.dt = float(mppi_cfg["dt"])
        self.ctrlTs = self.dt
        self.time_horizon = float(mppi_cfg["time_horizon"])
        self.horizon_length =  int(self.time_horizon//self.dt)

        self.mppi_num_rollouts = int(mppi_cfg["num_rollouts"])
        self.mppi_iteration_count = int(mppi_cfg["iteration_count"])

        self.is_close_loop = bool(mppi_cfg["is_close_loop"])
        self.do_timeshift = bool(mppi_cfg["do_timeshift"])

        self.robot_r = float(mppi_cfg["robot_radius"])

        self.add_emergency_maneuver = bool(mppi_cfg["add_emergency_maneuver"])
        
        #self.param_exploration = 0.2
        # -----------------------
        # Sampling / optimization config
        # -----------------------
        sampling_cfg = cfg["sampling"]
        self.sampling_type = str(sampling_cfg["type"])
        self.temperature = float(sampling_cfg["temperature"])
        self.temperature_l_bound = float(sampling_cfg["temperature_bounds"]["lower"])
        self.temperature_u_bound = float(sampling_cfg["temperature_bounds"]["upper"])

        self.knot_scale = int(sampling_cfg["knot_scale"])
        self.degree = int(sampling_cfg["spline_degree"])

        self.add_gamma_cost = bool(sampling_cfg["add_gamma_cost"])
        self.gamma_weight = float(sampling_cfg["gamma_weight"])

        self.is_update_temperature = bool(sampling_cfg["update_temperature"])

        # -----------------------
        # Risk / obstacle config
        # -----------------------
        risk_cfg = cfg["risk"]

        self.num_monte_carlo_samples = int(risk_cfg["num_monte_carlo_samples"])
        self.obs_unimodel_std_dev = float(risk_cfg["obs_unimodel_std_dev"])
        self.obs_w_soft = float(risk_cfg["obs_weights"]["soft"])
        self.obs_w_hard = float(risk_cfg["obs_weights"]["hard"])
        self.obs_sigma_threshold = float(risk_cfg["sigma_threshold"])

        # obstacle config
        obs_config = cfg["obstacle"]
        self.obs_r = float(obs_config["obstacle_radius"])
        self.num_obs = int(obs_config["num_obs"])

        # -----------------------
        # Random keys
        # -----------------------
        self.key = mppi_key
        self.key, self.control_pert_key = jax.random.split(self.key, 2)

        #-------------------
        # Load Model
        #-------------------
        self.model  = DifferentialDrive(mppi_cfg)
        # -----------------------
        # Control limits 
        # -----------------------
        self.ctrl_limit = jnp.asarray(mppi_cfg["control_limit"])
        
        # -----------------------
        # Trajectory-smoother
        # -----------------------
        self.trajectory_smoother = TrajectorySmoother(cfg["smoother"],self.dim_ctrl)

        # -----------------------
        # Critics
        # -----------------------
        self._load_critics(cfg["critics"],cfg["risk"])

        # -----------------------
        # Finish initialization
        # -----------------------
        self._init_mppi()


    # =====================================================
    # Critics
    # =====================================================
    def _load_critics(self, critics_cfg, risk_cfg):

        self.goal_critic = GoalCritic(critics_cfg["GoalCritic"],self.dim_euclid)
        self.goal_angle_critic = GoalAngleCritic(critics_cfg["GoalAngleCritic"],self.dim_euclid)
        self.obstacle_critic = DynObstaclesCritic(critics_cfg["ObstaclesCritic"], risk_cfg, self.dim_euclid)
        self.prefer_fwd_critic = PreferForwardCritic(critics_cfg["PreferForwardCritic"],self.dim_euclid,self.ctrlTs)
        self.path_angle_critic = PathAngleCritic(critics_cfg["PathAngleCritic"],self.dim_euclid)

    # =====================================================
    # MPPI internals
    # =====================================================
    def _init_mppi(self):
        #std-deviation of the perturbation k*sigma = control_limit 
        #if k=1 sigma = control_limit ; 68% of samples within the control-limit
        #k=2 95% of samples within the control-limit
        self.k = 1.5
        self.control_cov = jnp.diag(
            jnp.array([
                (self.ctrl_limit[0, 1] / self.k) ** 2,
                (self.ctrl_limit[1, 1] / self.k) ** 2
            ])
        )
        self.inv_control_cov = jnp.linalg.inv(self.control_cov)
        self.control_mean =  jnp.zeros((2,1))
        
        self.n_knots = self.horizon_length // self.knot_scale
        self.ndims = self.n_knots * self.dim_ctrl

        # Halton 
        # self.sequencer = ghalton.GeneralizedHalton(
        #     int(self.ndims),
        #     int(self.control_pert_key[0])
        # )       
        self.u_seqs = jnp.zeros((self.horizon_length, self.dim_ctrl))

    
    def prepare(self,rbt_curr_st,rbt_curr_vel,goal_pose,pred_obs_positions,pred_obs_vels):
        # dynamic risk aware mppi
        self.rbt_curr_pose = jnp.asarray(rbt_curr_st.reshape((self.dim_st,1)))
        if(self.do_timeshift):
           self.u_seqs = self.timeShift(self.u_seqs) # shift the control to the left, i.e receding horizon 
        if(self.is_close_loop):
            self.rbt_curr_vel  = jnp.asarray(rbt_curr_vel.reshape((self.dim_ctrl,1)))
        else:
            self.rbt_curr_vel =self.u_seqs[0,:].reshape((self.dim_ctrl,1))
            #previous_control = jnp.concatenate([curr_vel.transpose(),self.u_seqs[1:]], axis=0)
        self.prev_control_sequence =self.u_seqs
        self.rbt_goal_pose = jnp.asarray(goal_pose.reshape((self.dim_st,1)))
        self.pred_obs_positions = jnp.asarray(pred_obs_positions.reshape((self.num_obs,2))) #mean
        self.pred_obs_vels = jnp.asarray(pred_obs_vels.reshape((self.num_obs,2)))
        self.pred_obs_cov = jnp.broadcast_to(
            jnp.eye(2) * self.obs_unimodel_std_dev**2,
            (self.num_obs, 2, 2)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def solve(self,key,temperature,rbt_curr_pose,rbt_curr_vel,prev_control_sequence,rbt_goal_pose,pred_obs_positions,pred_obs_cov,pred_obs_vels):
        #@ compute the delta-u perturbations
        key,delta_u = self.control_pertubations(control_mean=self.control_mean, control_cov= self.control_cov,key = key)
        #@ compute the num_mppi_rolloutx1 costs
        key,nominal_mppi_cost , perturbed_control , delta_u, mppi_trajs = self.cal_nominal_mppi_cost(key,rbt_curr_pose,rbt_curr_vel,rbt_goal_pose,prev_control_sequence, pred_obs_positions, pred_obs_cov, pred_obs_vels,delta_u)
        total_cost = nominal_mppi_cost 
        if(self.add_gamma_cost):
            # prev_control_seq  (Tx2)
            # preturbed_control (KxTx2)
            #inv_control_cov (2x2)
            prod = jnp.einsum('ti,ij,ktj->kt',prev_control_sequence,self.inv_control_cov,delta_u)
            total_cost = total_cost + self.gamma_weight*jnp.sum(prod,axis=1)

        wght ,eta  = self.compute_weights(total_cost,temperature) # wght:(num_rollout,)
        temp =  jnp.sum(wght[:,None,None]*delta_u,axis=0)  #weighted delta-controls  ; delta: (num_rolloutxTxdim_ctrl) ; temp : (Txdim_ctrl)
        prev_control_sequence = prev_control_sequence+temp   # next-control += prev_control + delta-control                            
        
        # rbt_state = jnp.broadcast_to(
        #     rbt_curr_pose[None,:,0],
        #     (self.mppi_num_rollouts, self.dim_st)
        # )
        # # vel == control
        # rbt_vel = jnp.broadcast_to(
        #             rbt_curr_vel[None,:,0],
        #             (self.mppi_num_rollouts,self.dim_ctrl)
        #             )
       
        prev_control_sequence = self.model.batched_acc_limit_filter(rbt_curr_pose,rbt_curr_vel[None,:,0],prev_control_sequence[None,:,:],self.ctrlTs)
        prev_control_sequence = jnp.squeeze(prev_control_sequence,axis=0)
        #print("in solve " , prev_control_sequence.shape)
        #prev_control_sequence = self.control_clip(prev_control_sequence)  # clip the controls between the limits
        prev_control_sequence = self.trajectory_smoother(prev_control_sequence)
        return key,eta, prev_control_sequence , mppi_trajs
    
    def update_temperature(self,eta):
        # update the temperature-parameter online
        # https://arxiv.org/pdf/2307.09105  
        if self.is_update_temperature:
            if eta > self.temperature_u_bound:
                self.temperature = 0.8*self.temperature
        elif eta < self.temperature_l_bound:
            self.temperature = 1.2*self.temperature

    def get_optimal_seq(self,rbt_state,u_t):
        rbt_state = self.model.dynamics_step(rbt_state,u_t,self.ctrlTs)
        return rbt_state,rbt_state


    def compute_control(self,rbt_curr_st,rbt_curr_vel,goal_pose,pred_obs_poses,pred_obs_vels):
        
        self.prepare(rbt_curr_st,rbt_curr_vel,goal_pose,pred_obs_poses,pred_obs_vels)
        
        self.key, eta, self.u_seqs, mppi_trajs = self.solve(self.key,self.temperature,self.rbt_curr_pose,self.rbt_curr_vel,self.prev_control_sequence,self.rbt_goal_pose, self.pred_obs_positions, self.pred_obs_cov, self.pred_obs_vels)
    
        self.update_temperature(eta)

        optimal_control_t  = self.u_seqs[0,0:].reshape((self.dim_ctrl,1))        
        
        # Computing the optimal trajectory
        # loop through horizon-length
        _,X_optimal_seq = jax.lax.scan(self.get_optimal_seq,rbt_curr_st,self.u_seqs)
        
        
        optimal_seq_np = np.array(X_optimal_seq)
        X_rollout_np = np.array(mppi_trajs)
        optimal_control_t_np = np.array(optimal_control_t)
        return optimal_control_t_np, optimal_seq_np, X_rollout_np.transpose(1,0,2)
    

    def compute_weights(self,S:jnp.ndarray,temperature) -> jnp.ndarray:
        "compute  weights for each rollout"
        #prepare buffer
        rho = jnp.min(S)
        #print(rho)
        numerator = jnp.exp( (-1.0/temperature) * (S-rho) )
        #calculate the denominator , normalizer
        eta = jnp.sum(numerator)
        #calculate weight
        wt = (1.0 / eta) * numerator
        return wt ,eta  
    
    def batched_obstacle_step(self, pos, obstacle_cov, vel, dt):
        return pos + (vel ) * dt,  obstacle_cov + self.obs_unimodel_std_dev*dt*dt
    
    
    def make_step_fn(self,predicted_obs_vel,rbt_goal_pose):
        def step_function(carry,inputs):
            rbt_state,obstacles_mean_state, obstacle_cov, cost,goal_flag, key  = carry
            u_t = inputs

            # robot-step
            rbt_state = self.model.batched_dynamics_step(rbt_state,u_t,self.ctrlTs)

            #obstacles-step
    
            obstacles_mean_state,obstacle_cov = self.batched_obstacle_step(obstacles_mean_state,obstacle_cov,predicted_obs_vel,self.ctrlTs)

            #cost 
            subkey, key = jax.random.split(key,num=2)
            stage_cost, goal_flag = self.batched_stage_cost(rbt_state,u_t, goal_flag, obstacles_mean_state, obstacle_cov, rbt_goal_pose,subkey)
            cost = stage_cost+cost

            return (rbt_state,obstacles_mean_state,obstacle_cov,cost,goal_flag,key),rbt_state
        return step_function




    def cal_nominal_mppi_cost(self,key,rbt_curr_pose,rbt_curr_vel,rbt_goal_pose,prev_control_sequence,predicted_obs_state, obs_modelled_cov, predicted_obs_vel_array, delta_u):
        
        """
        Docstring for cal_nominal_mppi_cost
        
        :param self: Description
        :param key: Description
        :param rbt_curr_pose: Description (dim_stx1)
        :param rbt_curr_vel: Description (2x1)?
        :param rbt_goal_pose: Description
        :param prev_control_sequence: Description (Txdim_ctrl)
        :param predicted_obs_state: Description (mean obs state)
        :param obs_modelled_cov : Description ()
        :param predicted_obs_vel_array: Nox2
        :param delta_u: Description (num_rolloutxTxdim_ctrl)
        """        
        subkey,key = jax.random.split(key,2)
        # for batch rollout the initial state and previous control sequence is tiled for num_mppi_rollouts times

        #  param_exploration decides the percentage of exploration (adding control-noise) and  rest without noise, Not used in this simulation
        # ref : https://arxiv.org/abs/2209.12842
        # idx_exp : int = int((1-self.param_exploration)*self.mppi_num_rollouts) 

        # New control sequence is sum of previous(time-shifted ) computed optimal control + added gaussian noise
        u_seqs =  delta_u +  prev_control_sequence
        rbt_state = jnp.broadcast_to(
            rbt_curr_pose[None,:,0],
            (self.mppi_num_rollouts, self.dim_st)
        )
        # vel == control
        rbt_vel = jnp.broadcast_to(
                    rbt_curr_vel[None,:,0],
                    (self.mppi_num_rollouts,self.dim_ctrl)
                    )
        u_cliped_seqs = self.model.batched_acc_limit_filter(rbt_state,rbt_vel,u_seqs,self.ctrlTs)
        #u_cliped_seqs = self.control_clip(u_seqs)  #num_rolloutxTxdim_control
        
        if(self.add_emergency_maneuver):
            u_cliped_seqs = u_cliped_seqs.at[-1, 0:, :].set(0.0)
        
        #calculate stage and terminal costs ref, similar to : https://arxiv.org/abs/2210.00153
        goal_flag = jnp.zeros((self.mppi_num_rollouts,self.dim_euclid),dtype=jnp.float32) # (euclid,orientation)
        cost0 = jnp.zeros((self.mppi_num_rollouts,), dtype=jnp.float64)

        #obs_modelled_noise = jnp.zeros(shape=(self.horizon_length,self.num_obs, self.dim_euclid))
       
        init_carry = (rbt_state,predicted_obs_state, obs_modelled_cov, cost0, goal_flag, subkey)
        inputs = u_cliped_seqs.transpose(1,0,2)
        step_fn = self.make_step_fn(predicted_obs_vel=predicted_obs_vel_array,rbt_goal_pose=rbt_goal_pose)
        
        # loop through horizon-length
        carry_final,mppi_trajs = jax.lax.scan(step_fn,init_carry,inputs)


        rbt_state,obstacles_state,obs_cov,cost,goal_flag,key = carry_final
        
        # terminal_cost = self.batched_terminal_cost(rbt_state, goal_flag, obstacles_state,rbt_goal_pose)   
        # final_cost = terminal_cost+cost 
        
        delta_u = u_cliped_seqs-prev_control_sequence[None,:,:]
        return key,cost , u_cliped_seqs , delta_u , mppi_trajs 
          
    # defination of state-cost and terminal cost, similar to : https://arxiv.org/abs/2210.00153
    #equation 3,4,5,6 
    def batched_stage_cost(self, rbt_state, rbt_ctrl, goal_flag, obstacles_mean_state, obstacles_cov, rbt_goal_pose,key):       
        """
        Docstring for stage_cost
        
        :param self: Description
        :param rbt_state: Description (kx3)
        :param goal_flag: Description (kx2) (euclid-goal-flag, orient-goal-flag)
        :param obstacles_mean_state: Description (Nox2)
        :param obstacle_cov: modelled covariance (uncertainity) in the position (Nox2)
        :param goal_pose: Description (rbt_dim,1)
        """
        # eulcid distance to the obstacle
        cost_obs = self.obstacle_critic(rbt_state, obstacles_mean_state, obstacles_cov, rbt_goal_pose, key)



        #GoalCritic
        cost_to_goal  =    self.goal_critic(rbt_state,rbt_goal_pose)


        #GoalAngleCritic
        cost_to_goal_orient = self.goal_angle_critic(rbt_state,rbt_goal_pose)


        # new_goal_flag = jnp.stack([is_euclid_reached,is_reached_orient],axis=1)
        
        #Prefer Fwd Cost
        cost_to_prefer_fwd_dir = self.prefer_fwd_critic(rbt_state,rbt_ctrl,rbt_goal_pose)


        # Path Angle Critic
        cost_to_prefer_allign_angle = self.path_angle_critic(rbt_state,rbt_goal_pose)
        
        final_cost =   cost_obs+cost_to_prefer_fwd_dir+cost_to_prefer_allign_angle+cost_to_goal+cost_to_goal_orient

        return final_cost, goal_flag

    def batched_terminal_cost(self,rbt_state, goal_flag, obstacles_state, rbt_goal_pose):

        # eulcid distance to the obstacle
        rbt_xy = rbt_state[:,0:self.dim_euclid] #k,2

        #euclid goal error
        diff_to_goal      = rbt_xy - rbt_goal_pose[None,:self.dim_euclid,0]
        rbt_dist_to_goal  =  jnp.linalg.norm(diff_to_goal,axis=-1) #(k,)
        is_euclid_reached = ((rbt_dist_to_goal<=self.euclid_goal_tol) | (goal_flag[:,0] == 1.0)).astype(jnp.float32) #(k,)
        cost_to_goal      = (1-is_euclid_reached)*rbt_dist_to_goal*self.terminal_goal_cost_weight
    
        # orientation goal error
        rbt_yaw = rbt_state[:,-1]
        goal_yaw = rbt_goal_pose[None,-1,0]
        diff_yaw = rbt_yaw - goal_yaw 
        orient_error = jnp.arctan2(jnp.sin(diff_yaw),jnp.cos(diff_yaw))
        abs_orient_error = jnp.abs(orient_error)
        is_reached_orient  =  ((abs_orient_error<=self.goal_tolerance_orient) | (goal_flag[:,1]==1.0)).astype(jnp.float32) #(k,)
        cost_to_goal_orient = (1- is_reached_orient)*orient_error*self.terminal_goal_cost_weight_orient 

        #new_goal_flag = jnp.stack([is_euclid_reached,is_reached_orient],axis=1)
        
        final_cost =  cost_to_goal + cost_to_goal_orient
        return final_cost   
    
    

    
    
    def control_clip(self,v: jnp.ndarray):
        """
        v: (N, T, dim_ctrl)
        ctrl_limit: (dim_ctrl, 2) -> [min, max]
        """
        return jnp.clip(v, self.ctrl_limit[:, 0], self.ctrl_limit[:, 1])

    
    def temporal_smooth(self,noise, window=7):
        kernel = jnp.ones((window,)) / window
        kernel = kernel[:, None]  # (W,1)

        def smooth_1d(x):
            return jnp.convolve(x, kernel[:, 0], mode="same")

        return jax.vmap(
            jax.vmap(smooth_1d, in_axes=1, out_axes=1),
            in_axes=0
        )(noise)
        
    #https://arxiv.org/pdf/2307.09105  
    # # https://proceedings.mlr.press/v164/bhardwaj22a.html
    # idea : sample the noise from ghalton for uniform sampling between [0,1],
    # convert it to the gaussian noise using erfinv then use the B-spline to smooth out the control actions
    # for low dimension, a simple gaussian sampler could work
    def control_pertubations(self, control_mean, control_cov,key):

        # if self.sampling_type == "gaussian_halton":
        #     sample_shape = self.mppi_num_rollouts
            
        #     knot_points = np.array(self.sequencer.get(self.mppi_num_rollouts),dtype=float)
        #     # map uniform Halton points → standard normal samples
        #     gaussian_halton_samples = np.sqrt(2.0)*scsp.erfinv(2 * knot_points - 1)
        #     # Sample splines from knot points:
        #     # iteratre over action dimension:
        #     # reshape to (rollouts, dim_ctrl, n_knots)
        #     knot_samples = gaussian_halton_samples.reshape(self.mppi_num_rollouts, self.dim_ctrl, self.n_knots) # n knots is T/knot_scale 
        #     delta_u = np.zeros((sample_shape, self.horizon_length, self.dim_ctrl))
          
        #     for i in range(sample_shape):
        #         for j in range(self.dim_ctrl):
        #             delta_u[i,:,j] = self.bspline(knot_samples[i,j,:], n=self.horizon_length, degree=self.degree)
        #     temp =  np.sqrt((control_cov))
        #     delta_u = np.matmul(delta_u ,temp )
        #     return key,delta_u    
        # elif self.sampling_type == "jax_gaussian":
        key, subkey = jax.random.split(key)
        L = jnp.linalg.cholesky(control_cov)
        eps = jax.random.normal(subkey, (self.mppi_num_rollouts, self.horizon_length, self.dim_ctrl),dtype=jnp.float32)
        # temporal smoothing
        #eps = self.temporal_smooth(eps)
        delta_u = eps @ L.T + control_mean[:,0]
        return key, delta_u


    def moving_average_filter(self,xx: jnp.ndarray, window_size: int) -> jnp.ndarray:
        """
        apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        xx: (T, dim)
        window_size: int
        """
        T, dim = xx.shape

        # reshape input to (N=1, W=T, C=dim)
        x = xx[None, :, :]

        # depthwise kernel: (KW, 1, C)
        kernel = (
            jnp.ones((window_size, 1, dim), dtype=xx.dtype)
            / window_size
        )

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=dim
        )
        return y[0]


    @functools.partial(jax.jit, static_argnums=0) 
    def timeShift(self,ctrl:jnp.array)->jnp.array:
        return jnp.vstack([ctrl[1:],ctrl[-1]])


    # def bspline(self,c_arr, t_arr=None, n=100, degree=3):
    #     #sample_device = c_arr.device
    #     sample_dtype = c_arr.dtype
    #     cv = c_arr

    #     if(t_arr is None):
    #         t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    #     else:
    #         t_arr = t_arr
    #     spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    #     xx = np.linspace(0, cv.shape[0], n)
    #     samples = si.splev(xx, spl, ext=3)
    #     samples = np.array(samples, dtype=sample_dtype)
    #     return samples  