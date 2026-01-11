#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools

class DynObstaclesCritic:
    """
    Docstring for DynObstaclesCritic
    Compute the minimum obstacle distance per rollout, 
    classify it into mutually exclusive regions, 
    and apply exactly one cost term per rollout.
    """
    def __init__(self, cfg, risk_cfg, euclid_dim):
        self.enabled = cfg["enabled"]
        self.power = cfg["cost_power"]
        self.repulsion_weight = cfg["repulsion_weight"]
        self.critical_weight = cfg["critical_weight"]
        self.collision_cost = cfg["collision_cost"]
        self.collision_margin_distance = cfg["collision_margin_distance"]
        self.near_goal_distance = cfg["near_goal_distance"]
        self.robot_r = cfg["robot_radius"]
        self.obs_r = cfg["obstacles_radius"]
        self.buffer_r = cfg["buffer_radius"]
        self.consider_obs_density = cfg["consider_obs_density"]
        self.effective_radius = self.robot_r+self.obs_r
        self.euclid_dim = euclid_dim

        self.num_monte_carlo_samples = int(risk_cfg["num_monte_carlo_samples"])
        self.obs_unimodel_std_dev = float(risk_cfg["obs_unimodel_std_dev"])
        self.obs_w_soft = float(risk_cfg["obs_weights"]["soft"])
        self.obs_w_hard = float(risk_cfg["obs_weights"]["hard"])
        self.obs_sigma_threshold = float(risk_cfg["sigma_threshold"])
    
    @functools.partial(jax.jit, static_argnums=0)
    def obstacle_pdfs(self,mc_points,means,covs):
        """
        Docstring for obstacle_pdfs
        
        :param self: Description
        :param mc_points: Nmc*2
        :param means: No*2
        :param covs: No*(2*2)
        :returns p[0_{j}] = p_t^0(x_j,y_j)
        :shape: (No, mc_points)
        """
        diff = mc_points[:,None,:] - means[None,:,:] # (Nmc,No,2)
        inv_cov = jnp.linalg.inv(covs) #(No,2,2)
        # No,Nmc
        quad = jnp.einsum('moi,oij,moj->om',diff,inv_cov,diff)
        #Normalization constant (No,)
        det_cov = jnp.linalg.det(covs)
        norm = 1.0/(2.0*jnp.pi*jnp.sqrt(det_cov))
        # pdf (No,Nmc)
        pdfs = norm[:,None]*jnp.exp(-0.5*quad)
        return pdfs

    @functools.partial(jax.jit,static_argnums = 0 )
    def collision_probability(self,p_obs_density,minval,maxval):
        """
            Docstring for joint_collision_probability
            :param self: Description
            :param p_obs: No*Nmc 
            :returns p_joint Nmc,
        """
        num_points = p_obs_density.shape[0]
        box_area = (maxval[0]-minval[0])*(maxval[1]-minval[1])
        delta_A = box_area/num_points

        # convert density - > probability
        p_obs = p_obs_density*delta_A
        # safety clamp
        p_obs = jnp.clip(p_obs,0.0,1.0-1e-6)
        
        # Probability of collision at Nmc points
        p_joint = 1 - jnp.exp(jnp.sum(jnp.log1p(-p_obs),axis=0))
        return p_joint
    @functools.partial(jax.jit,static_argnums=0)
    def joint_coll_prob_with_robot(self,
                                   p_cp,  #(Nmc,)
                                   monte_carlo_points, # (Nmc, 2)
                                   rbt_xy  # (K, 2)
                                   ):

        #diff (K,Nmc,2)
        diff = monte_carlo_points[None,:,:] - rbt_xy[:,None,:] #(None,Nmc,2) - (k,None,2)
        
        #squared-distance
        disk_sq = jnp.sum(diff**2,axis=2)
        mask = disk_sq <= (self.effective_radius**2)  # (k,Nmc)

        #number of mc points inside disk
        n_inside = jnp.sum(mask,axis=1) #(k,)
        n_inside = jnp.maximum(n_inside,1) # to avoid overflow
        p_joint_cp_with_rbt = (jnp.pi*self.effective_radius**2 * 
                               jnp.sum(mask*p_cp[None,:],axis=1)/n_inside
                               ) # (k,)

        return p_joint_cp_with_rbt
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, state, obs_state, obs_cov, goal, key):
        euclid_dim = self.euclid_dim
        if not self.enabled:
            return jnp.zeros((state.shape[0],))
        if self.consider_obs_density:
            # Being surrounded by many obstacles is more dangerous 
            # than being near just one.
            # -----------------------
            # Goal relaxation
            # -----------------------
            dist_to_goal = jnp.linalg.norm(
                state[:, :euclid_dim] - goal[:euclid_dim,0], axis=1
            )  # (K,)
            relax_mask = dist_to_goal < self.near_goal_distance

            # -----------------------
            # Distances to obstacles
            # -----------------------
            rbt_xy = state[:, :euclid_dim]          # (K, 2)



            x_min = jnp.min(rbt_xy[:,0]) -  self.effective_radius 
            x_max = jnp.max(rbt_xy[:,0]) +  self.effective_radius
            y_min = jnp.min(rbt_xy[:,1]) -  self.effective_radius
            y_max = jnp.max(rbt_xy[:,1]) +  self.effective_radius
            monte_carlo_obs_samples = jax.random.uniform(
                key=key,
                shape=(self.num_monte_carlo_samples, self.euclid_dim),
                minval=jnp.array([x_min, y_min]),
                maxval=jnp.array([x_max, y_max])
            )

            p_obs = self.obstacle_pdfs(monte_carlo_obs_samples, obs_state, obs_cov)   # (No, Nmc)
            # Collision probability for  monte_carlo_obs_samples samples 
            # Nmc,
            p_cp = self.collision_probability(p_obs,minval=jnp.array([x_min, y_min]), maxval=jnp.array([x_max, y_max]))
            # coliision probability at the k, robot positions 
            
            p_joint_cp = self.joint_coll_prob_with_robot(p_cp,monte_carlo_obs_samples,rbt_xy)

            risk_cost_obs  = self.obs_w_soft*p_joint_cp + self.obs_w_hard*(p_joint_cp  > self.obs_sigma_threshold) #equation-14 # per time step collision probability


            # obs_xy = obs_state[:, :euclid_dim]      # (N, 2)

            # diff = rbt_xy[:, None, :] - obs_xy[None, :, :]   # (K, N, 2)
            # dists = jnp.linalg.norm(diff, axis=-1)           # (K, N)

            # r_eff = self.effective_radius
            # delta = self.collision_margin_distance

            # # -----------------------
            # # Region masks (K, N)
            # # -----------------------
            # collision_mask = dists < r_eff
            # critical_mask  = (dists >= r_eff) & (dists < r_eff + delta)
            # soft_mask      = dists >= (r_eff + delta)

            # # -----------------------
            # # Per-obstacle costs
            # # -----------------------
            # collision_cost = collision_mask * self.collision_cost

            # critical_cost = critical_mask * (
            #     self.critical_weight *
            #     jnp.exp(-self.power * (dists - r_eff))
            # )

            # soft_cost = soft_mask * (
            #     self.repulsion_weight *
            #     jnp.exp(-self.power * (dists - r_eff - delta))
            # )

            # # -----------------------
            # # Sum over obstacles
            # # -----------------------
            # total_cost = jnp.sum(
            #     collision_cost + critical_cost + soft_cost,
            #     axis=1
            # )  # (K,)

            # -----------------------
            # Near-goal relaxation
            # -----------------------
            #total_cost = jnp.where(relax_mask, 0.3 * total_cost, total_cost)

            return risk_cost_obs


