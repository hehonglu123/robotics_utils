from general_robotics_toolbox import * 
from general_robotics_toolbox import robotraconteur as rr_rox

import numpy as np
import pickle
from robotics_utils import *


class robot_obj(object):
	###robot object class
	def __init__(self,robot_name,def_path,tool_file_path='',base_transformation_file='',acc_dict_path='',d=0):
		#def_path: robot 			definition yaml file, name must include robot vendor
		#tool_file_path: 			tool transformation to robot flange csv file
		#base_transformation_file: 	base transformation to world frame csv file
		#acc_dict_path: 			accleration profile

		self.robot_name=robot_name
		with open(def_path, 'r') as f:
			self.robot = rr_rox.load_robot_info_yaml_to_robot(f)

		self.def_path=def_path
		#define robot without tool
		self.robot_def_nT=Robot(self.robot.H,self.robot.P,self.robot.joint_type)

		if len(tool_file_path)>0:
			tool_H=np.loadtxt(tool_file_path,delimiter=',')
			self.robot.R_tool=tool_H[:3,:3]
			self.robot.p_tool=tool_H[:3,-1]+np.dot(tool_H[:3,:3],np.array([0,0,d]))
			self.p_tool=self.robot.p_tool
			self.R_tool=self.robot.R_tool		

		if len(base_transformation_file)>0:
			self.base_H=np.loadtxt(base_transformation_file,delimiter=',')
		else:
			self.base_H=np.eye(4)


		###set attributes
		self.upper_limit=self.robot.joint_upper_limit 
		self.lower_limit=self.robot.joint_lower_limit 
		self.joint_vel_limit=self.robot.joint_vel_limit 
		self.joint_acc_limit=self.robot.joint_acc_limit 

		###acceleration table
		if len(acc_dict_path)>0:
			acc_dict= pickle.load(open(acc_dict_path,'rb'))
			q2_config=[]
			q3_config=[]
			q1_acc_n=[]
			q1_acc_p=[]
			q2_acc_n=[]
			q2_acc_p=[]
			q3_acc_n=[]
			q3_acc_p=[]
			for key, value in acc_dict.items():
				q2_config.append(key[0])
				q3_config.append(key[1])
				q1_acc_n.append(value[0%len(value)])
				q1_acc_p.append(value[1%len(value)])
				q2_acc_n.append(value[2%len(value)])
				q2_acc_p.append(value[3%len(value)])
				q3_acc_n.append(value[4%len(value)])
				q3_acc_p.append(value[5%len(value)])
			self.q2q3_config=np.array([q2_config,q3_config]).T
			self.q1q2q3_acc=np.array([q1_acc_n,q1_acc_p,q2_acc_n,q2_acc_p,q3_acc_n,q3_acc_p]).T
		

	def get_acc(self,q_all,direction=[]):
		###get acceleration limit from q config, assume last 3 joints acc fixed direction is 3 length vector, 0 is -, 1 is +
		#if a single point
		if q_all.ndim==1:
			###find closest q2q3 config, along with constant last 3 joints acc
			idx=np.argmin(np.linalg.norm(self.q2q3_config-q_all[1:3],axis=1))
			acc_lim=[]
			if len(direction)==0:
				raise AssertionError('direciton not provided')
				return
			for d in direction:
				acc_lim.append(self.q1q2q3_acc[idx][2*len(acc_lim)+d])

			return np.append(acc_lim,self.joint_acc_limit[-3:])
		#if a list of points
		else:
			dq=np.gradient(q_all,axis=0)[:,:3]
			direction=(np.sign(dq)+1)/2
			direction=direction.astype(int)
			acc_limit_all=[]
			for i in range(len(q_all)):
				idx=np.argmin(np.linalg.norm(self.q2q3_config-q_all[i][1:3],axis=1))
				acc_lim=[]
				for d in direction[i]:
					acc_lim.append(self.q1q2q3_acc[idx][2*len(acc_lim)+d])

				acc_limit_all.append(np.append(acc_lim,self.joint_acc_limit[-3:]))

		return np.array(acc_limit_all)

	def fwd(self,q_all,world=False,qlim_override=False):
		###robot forworld kinematics
		#q_all:			robot joint angles or list of robot joint angles
		#world:			bool, if want to get coordinate in world frame or robot base frame

		if q_all.ndim==1:
			q=q_all
			pose_temp=fwdkin(self.robot,q, _ignore_limits = qlim_override)

			if world:
				pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
				pose_temp.R=self.base_H[:3,:3]@pose_temp.R
			return pose_temp
		else:
			pose_p_all=[]
			pose_R_all=[]
			for q in q_all:
				pose_temp=fwdkin(self.robot,q, _ignore_limits = qlim_override)
				if world:
					pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
					pose_temp.R=self.base_H[:3,:3]@pose_temp.R

				pose_p_all.append(pose_temp.p)
				pose_R_all.append(pose_temp.R)

			return Transform_all(pose_p_all,pose_R_all)
	
	def jacobian(self,q):
		return robotjacobian(self.robot,q)

	def inv(self,p,R=np.eye(3),last_joints=None):
		pose=Transform(R,p)
		q_all=robot6_sphericalwrist_invkin(self.robot,pose,last_joints)
		
		return q_all
	
	def iter_inv(self,p,R,last_joints):
		pose=Transform(R,p)
		converged, q_normed = iterative_invkin(self.robot,pose,last_joints)
		if converged:
			return np.array(q_normed[0])
		else:
			raise Exception('Inverse kinematics did not converge')
	
	###find a continous trajectory given Cartesion pose trajectory
	def find_curve_js(self,curve,curve_R,q_seed=None):
		q_inits=self.inv(curve[0],curve_R[0])
		curve_js_all=[]
		for q_init in q_inits:
			curve_js=np.zeros((len(curve),6))
			curve_js[0]=q_init
			for i in range(1,len(curve)):
				q_all=np.array(self.inv(curve[i],curve_R[i]))
				if len(q_all)==0:
					#if no solution
					print('no solution available')
					return

				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				if np.linalg.norm(q_all[order[0]]-curve_js[i-1])>0.5:
					break	#if large changes in q, not continuous
				else:
					curve_js[i]=q_all[order[0]]

			#check if all q found
			if i==len(curve)-1:
				curve_js_all.append(curve_js)
		
		if len(curve_js_all)==0:
			raise Exception('No Solution Found') 
		
		if q_seed is None:
			return curve_js_all
		else:
			if len(curve_js_all)==1:
				return curve_js_all[0]
			else:
				diff_min=[]
				for curve_js in curve_js_all:
					diff_min.append(np.linalg.norm(curve_js[0]-q_seed))

				return curve_js_all[np.argmin(diff_min)]


class Transform_all(object):
	def __init__(self, p_all, R_all):
		self.R_all=np.array(R_all)
		self.p_all=np.array(p_all)

