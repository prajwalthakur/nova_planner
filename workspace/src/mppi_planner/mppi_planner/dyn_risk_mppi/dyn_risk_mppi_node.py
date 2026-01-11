#!/usr/bin/env python3
import numpy as np
import yaml
import time
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # 0.9 causes too much lag. 
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import jax
import jax.numpy as jnp
from jax import config  # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

from .dyn_risk_mppi_class import MPPI
from mppi_planner.config_loader import  configLoader 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path, Odometry
from tf_transformations import euler_from_quaternion  
from dataclasses import dataclass
from typing import Dict


# Load core parameters from YAML
config_path  = "src/mppi_planner/config/dyn_sim_config.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

seed                     = int(cfg['seed'])
dt                       = float(cfg['mppi']['dt'])
dim_ctrl                 = int(cfg['dim_control'])
euclid_goal_tol          = float(cfg['euclid_goal_tol'])
orient_goal_tol          =  float(cfg['orient_goal_tol'])
time_horizon           = int(cfg['time_horizon'])
horizon_length = int(time_horizon//dt)
mppi_num_rollouts        = int(cfg['mppi']['num_rollouts'])
pose_lim                 = jnp.array(cfg['pose_lim'])
num_obs                  = int(cfg['num_obs'])



# class GaussianNoiseSampler:
#     def __init__(self, mean, stddev: float, seed: int, dim: int):
#         self.stddev = stddev
#         self.key = jax.random.PRNGKey(seed)
#         self.dim_ = dim
#         self.mean = mean

#     def sample(self):
#         self.key, subkey = jax.random.split(self.key)
#         samples = self.mean + jax.random.normal(subkey, shape=(self.dim_,)) * self.stddev
#         return samples

# def sample_uniform(key, min_val, max_val, num_samples):
#     return jax.random.uniform(
#         key,
#         shape=(num_samples,),
#         minval=min_val,
#         maxval=max_val
#     )
# def set_random_goal_by_agent(key, current_state, current_goal , goal_threshold, pose_limit):
#     minVal = pose_limit[0]
#     maxVal = pose_limit[1]
#     current_state.reshape(3,1)
#     current_goal.reshape(3,1)
#     if(np.linalg.norm(current_state[0:2] - current_goal[0:2] ) < goal_threshold):
#         # goal reached 
#         key,subkey = jax.random.split(key,2)
#         newXGoal = sample_uniform(subkey,minVal[0],maxVal[0],1)
#         key,subkey = jax.random.split(key,2)
#         newYGoal = sample_uniform(subkey,minVal[1],maxVal[1],1)
#         key,subkey = jax.random.split(key,2)
#         newYawGoal = sample_uniform(subkey,0.0,np.pi,1)
#         return (key,jnp.array([newXGoal,newYGoal,newYawGoal]).reshape(3,1))
#     return (key,current_goal)      


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    array_index: int = 0

class SimplePlanner(Node):
    def __init__(self):
        super().__init__('vanilla_mppi_planner_node')
        self.loaded_config = configLoader(path=config_path)
        self.init_member_var()
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.robot_odom_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        # visualization utility
        self.path_pub = self.create_publisher(Path,  '/mppi/robot_path',   10)
        self.opt_pub  = self.create_publisher(Marker,  '/mppi/opt_rollout',     10)
        self.roll_pub = self.create_publisher(Marker,'/mppi/rollouts',     10)
        self.robot_path = Path()
        self.robot_path.header.frame_id = 'odom'

        for i in range(num_obs):
            name = f"obstacle_{i}"

            self.obs_subs[name] = self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, n=name: self.obs_odom_callback(msg, n),
                10
            )

            st = State()
            st.array_index = i
            self.obs_states_[name] = st

    
    def init_member_var(self):
        
        self.scene_pose_limit = pose_lim
        self.euclid_goal_tol = euclid_goal_tol
        self.orient_goal_tol = orient_goal_tol
        self.obs_subs = {}          # subscriptions live here
        self.obs_states_ = {}      # State structs live here

        self.obs_states_array = np.zeros((num_obs,2))
        self.obs_states_vel_array = np.zeros((num_obs,2))
        self.rbt_current_pose  =  np.zeros((3,)) 
        self.rbt_current_vel = np.zeros((2,)) 
        self.timer_period = dt  # 1/(planning rate) currently ~ 10hz
        # timer to call planning function
        self.control_timer = self.create_timer(self.timer_period, self.control_cb)
        # Create a deterministic PRNGKey from our fixed seed so that MPPI sampling is reproducible.
        self.global_key = jax.random.PRNGKey(seed)
        self.init_mppi = False

    def goal_callback(self, msg: PoseStamped):
        """Updates the goal from RViz."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Update the internal goal pose (using (3, 1) shape to match your MPPI logic)
        self.goal_pose = jnp.array([x, y, yaw]).reshape(3, 1)
        
        self.get_logger().info(f"New goal received from RViz: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
    def obs_odom_callback(self, msg: Odometry , name : str):
        st = self.obs_states_[name]
        curr_position = msg.pose.pose.position
        curr_orient = msg.pose.pose.orientation
        st.vx = msg.twist.twist.linear.x
        st.vy = msg.twist.twist.linear.y
        quaternion = [curr_orient.x, curr_orient.y, curr_orient.z, curr_orient.w]
        # Convert to Euler angles
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        st.x = curr_position.x
        st.y = curr_position.y
        st.z = curr_position.z
        st.yaw = yaw
        idx = st.array_index
        self.obs_states_array[idx, :] = np.array([st.x, st.y])
        self.obs_states_vel_array[idx, :] = np.array([st.vx, st.vy])

    def robot_odom_callback(self, msg: Odometry):
        """Callback to handle incoming odometry messages."""
        current_pose = msg.pose.pose
        q = current_pose.orientation
        vx = msg.twist.twist.linear.x
        angualarZ = msg.twist.twist.angular.z
        quaternion = [q.x, q.y, q.z, q.w]

        # Convert to Euler angles
        _, _, yaw = euler_from_quaternion(quaternion)
        # self.get_logger().debug(
        #     f'Received odom: position=({current_pose.position.x:.2f}, '
        #     f'{current_pose.position.y:.2f})'
        # )
        self.rbt_current_pose  = np.array([current_pose.position.x,current_pose.position.y,yaw])
        self.rbt_current_vel = np.array([vx,angualarZ])
        # Split global key into two independent keys:
        #  - mppi_key: used for generating control perturbations
        #  - goal_key: reserved for any goal‐related randomness
        if self.init_mppi is False:
            self.global_key,key  = jax.random.split(self.global_key, 2)
            self.goal_pose = self.rbt_current_pose
            self.MppiObj  = MPPI(self.loaded_config, key)
            self.init_mppi = True
  
    def is_goal_reached(self, current_pose, goal_pose):
        euclid_dist = jnp.linalg.norm(current_pose[0:2] - goal_pose[0:2])
        orient_dist = jnp.abs(jnp.arctan2(
                    jnp.sin(current_pose[2] - goal_pose[2]),
                    jnp.cos(current_pose[2] - goal_pose[2]))
                    )
        if((euclid_dist < self.euclid_goal_tol) and (orient_dist < self.orient_goal_tol)):
            return euclid_dist,orient_dist,True
        return euclid_dist,orient_dist,False

    def control_cb(self):
        twist = Twist()
        print("in control cb")
        X_optimal_seq = np.zeros((horizon_length,dim_ctrl))
        X_rollout = np.zeros((mppi_num_rollouts,horizon_length,dim_ctrl))
        if self.init_mppi is True:
            #key, current_state, current_goal , goal_threshold, pose_limit
            #self.global_key, self.goal_pose = set_random_goal_by_agent(key=self.global_key,current_state=self.rbt_current_pose,current_goal=self.goal_pose,goal_threshold=self.euclid_goal_tol,pose_limit=self.scene_pose_limit)
            # dist_to_goal = jnp.linalg.norm(self.rbt_current_pose[0:2] - self.goal_pose[0:2])
            # if( dist_to_goal<= goal_tolerance):
            #     optimal_control = np.zeros((dim_ctrl,1))
            #else:
            start = time.time()
            optimal_control, X_optimal_seq,X_rollout = self.MppiObj.compute_control(self.rbt_current_pose,self.rbt_current_vel,self.goal_pose,self.obs_states_array,self.obs_states_vel_array)
            self.get_logger().info(f'time took to compute control commands ={time.time()-start}')
            euclid_dist,orient_dist,isReached = self.is_goal_reached(self.rbt_current_pose,self.goal_pose)
            if(isReached):
                optimal_control = np.zeros((dim_ctrl,1))
            
            
            twist.linear.x = optimal_control[0][0]
            twist.angular.z = optimal_control[1][0]
            self.cmd_vel_pub.publish(twist)

            self.get_logger().info(f'Published cmd_vel: linear.x={twist.linear.x:.2f},angular.z={twist.angular.z:.2f}')
            self.get_logger().info(f'euclid dist to goal={euclid_dist:.2f} , orient-dist = {orient_dist}')
            self.publish_path_utils(X_optimal_seq,X_rollout,num_to_vis=20)
                
    # function to plot traced path, optimal predicted path and MPPI trajectory rollouts        
    def publish_path_utils(self, X_optimal_seq, X_rollout, num_to_vis=20):
        now = self.get_clock().now().to_msg()

        # 1) Robot’s past path
        self.robot_path.header.stamp = now
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = 'odom'
        ps.pose.position.x = float(self.rbt_current_pose[0])
        ps.pose.position.y = float(self.rbt_current_pose[1])
        ps.pose.orientation.w = 1.0
        self.robot_path.poses.append(ps)
        self.path_pub.publish(self.robot_path)

        
        opt_marker = Marker()
        opt_marker.header.stamp = now
        opt_marker.header.frame_id = 'odom'
        opt_marker.ns = 'mppi_optimal_rollout'
        opt_marker.id = 0
        opt_marker.type = Marker.LINE_LIST
        opt_marker.action = Marker.ADD
        opt_marker.scale.x = 0.02  
        opt_marker.color.r = 1.0
        opt_marker.color.g = 0.0
        opt_marker.color.b = 0.0
        opt_marker.color.a = 1.0        
        for i in range(X_optimal_seq.shape[0] - 1):
            p0 = X_optimal_seq[i,:]
            p1 = X_optimal_seq[i+1,:]
            opt_marker.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=0.0))
            opt_marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=0.0))        
        self.opt_pub.publish(opt_marker)

        #  MPPI rollouts as a Marker (LINE_LIST)
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = 'odom'
        marker.ns = 'mppi_rollouts'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.1

        # Only visualize num_to_vis rollouts
        step = int(X_rollout.shape[0]/num_to_vis)
        for itr in  range(0,X_rollout.shape[0],step):
            rollout = X_rollout[itr]
            # rollout is [H x 2], draw segments between consecutive points
            for i in range(rollout.shape[0] - 1):
                p0 = rollout[i]
                p1 = rollout[i+1]
                marker.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=0.0))
                marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=0.0))

        self.roll_pub.publish(marker)      

def main():
    rclpy.init()
    node = SimplePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
