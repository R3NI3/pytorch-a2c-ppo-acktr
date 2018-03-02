#import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import vrep
import os
import time
import numpy as np
from operator import sub

class VrepSoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ip = '127.0.0.1'
        port = 19997
        time_step = 0.01
        actuator_names = ['LeftMotor', 'RightMotor']
        object_names = ['Bola', 'Dummy']
        robot_names = ['DifferentialDriveRobot']
        scene_path = './Cenario.ttt'
         # Connect to the V-REP continuous server
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart(ip, port, True, True, -500000, 5)
        if self.clientID != -1: # if we connected successfully
            print ('Connected to remote API server')
        else:
            raise Exception()
        self._configure_environment(scene_path)
        # -------Setup the simulation
        vrep.simxSynchronous(self.clientID, True) #if we need to be syncronous
        # move simulation ahead one time step
        vrep.simxSynchronousTrigger(self.clientID)

        vrep.simxSetFloatingParameter(self.clientID,
                vrep.sim_floatparam_simulation_time_step,
                time_step, # specify a simulation time step
                vrep.simx_opmode_oneshot)

        self.get_handles(actuator_names, robot_names, object_names)

        self.time_step = time_step
        #todo check if observation_space and action_space are correct
        shape = len(robot_names)*5 + len(object_names)*4
        self.observation_space = spaces.Box(low=0, high=1700, shape=[shape])
        shape = len(robot_names)*2
        self.action_space = spaces.Box(low=-10, high=10, shape=[shape])

        self.getSimulationState()

        robot_pos = np.array(self.state['DifferentialDriveRobot'][0:2])
        target_pos = np.array(self.state['Bola'][0:2])
        self.old_distance = np.linalg.norm(robot_pos - target_pos)
        self.init_orientation = self.get_orientation('DifferentialDriveRobot')


    def __del__(self):
        vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)
        vrep.simxFinish(self.clientID)

    def _configure_environment(self, scene_path):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        vrep.simxLoadScene(self.clientID, scene_path, 1,vrep.simx_opmode_blocking)
        self._turn_display(False)

    def _turn_display(self, Value = True):
        if (Value):
            vrep.simxSetFloatingParameter(self.clientID, vrep.sim_boolparam_display_enabled, True, vrep.simx_opmode_blocking)
        else:
            vrep.simxSetFloatingParameter(self.clientID, vrep.sim_boolparam_display_enabled, False, vrep.simx_opmode_blocking)

    def step(self, action):
        self._take_action(action)
        state_info = self.getSimulationState()
        done = self.check_done()
        return state_info,self.get_reward(),done, {}

    def check_done(self):
        angles = self.get_orientation('DifferentialDriveRobot')
        bola_pos = self.get_position('Bola')
        angles_diff = list(map(sub, angles, self.init_orientation))
        if (abs(angles_diff[0]) > 20 or abs(angles_diff[1]) > 20):
            print("*******DONE1*****")
            self.reset()
            return True

        if (abs(bola_pos[2]) > 10):
            print("*******DONE2*****")
            self.reset()
            return True

        dummy_pos = np.array(self.state['Dummy'][0:2])
        target_pos = np.array(self.state['Bola'][0:2])
        distance = np.linalg.norm(dummy_pos - target_pos)
        if (distance < 0.2):
            print("******Done3******")
            self.reset()
            return True

        return False

    def _take_action(self, action):
        if (abs(action[0]) > 1):
            l = 1 if action[0] > 0 else -1
        else:
            l = action[0]
        l = l/10
        omega = action[1]
        HALF_AXIS = 0.0325
        WHEEL_R = 0.032/2
        motorSpeed = [0,0]
        #angularFreqL = l - HALF_AXIS * omega;
        #angularFreqR = l + HALF_AXIS * omega;
        #motorSpeedL = (angularFreqL / WHEEL_R);
        #motorSpeedR = (angularFreqR / WHEEL_R);
        angularFreqL = l - HALF_AXIS * omega;
        angularFreqR = l + HALF_AXIS * omega;
        motorSpeed[0] = (angularFreqL / WHEEL_R);
        motorSpeed[1] = (angularFreqR / WHEEL_R);

        handles = [self.act_handles[key] for key in sorted(self.act_handles.keys(), reverse=True)]
        for idx,motor_handle in enumerate(handles):
            vrep.simxSetJointTargetVelocity(self.clientID, motor_handle,
                        motorSpeed[idx], # target velocity
                        vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.clientID)

    def get_reward(self):
        state_info = self.state

        robot_pos = np.array(state_info['DifferentialDriveRobot'][0:2])
        target_pos = np.array(state_info['Bola'][0:2])
        dummy_pos = np.array(state_info['Dummy'][0:2])
        distance = np.linalg.norm(dummy_pos - target_pos)
        distance2 = np.linalg.norm(robot_pos - target_pos)
        if (distance < 0.2):
            reward = 100000
        else:
            reward = -1 + (self.old_distance - distance)*10 - distance2
            #reward = -2 if distance > self.old_distance else 1
            #reward = 0
        #reward = (self.old_distance - distance)*100 if distance < self.old_distance else -1
        self.old_distance = distance
        #reward -= distance*10#1/ distance if distance != 0 else 1000
        return reward

    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)
        self._turn_display(False)
        return self.getSimulationState()

    def render(self):
        pass

    def stop_robot(self, actuator_names):
        motor_handles = [self.act_handles[act_name] for act_name in actuator_names
                            if act_name in self.act_handles]
        for ii,motor_handle in enumerate(motor_handles):
            # if force has changed signs,
            # we need to change the target velocity sign
            vrep.simxSetJointTargetVelocity(self.clientID,
                        motor_handle,
                        0, # target velocity
                        vrep.simx_opmode_blocking)
        # move simulation ahead one time step
        vrep.simxSynchronousTrigger(self.clientID)

    def get_position(self, obj_name):
        if obj_name in self.ddr_handles:
            obj_handle = self.ddr_handles[obj_name]
        elif obj_name in self.obj_handles:
            obj_handle = self.obj_handles[obj_name]
        elif obj_name in self.act_handles:
            obj_handle = self.act_handles[obj_name]
        else:
            return -1

        _, obj_xyz = vrep.simxGetObjectPosition(self.clientID, obj_handle,
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_blocking)
        if _ !=0 : raise Exception()
        else: return obj_xyz

    def get_orientation(self, obj_name):
        if obj_name in self.ddr_handles:
            obj_handle = self.ddr_handles[obj_name]
        elif obj_name in self.obj_handles:
            obj_handle = self.obj_handles[obj_name]
        elif obj_name in self.act_handles:
            obj_handle = self.act_handles[obj_name]
        else:
            return -1

        _, obj_ang = vrep.simxGetObjectOrientation(self.clientID, obj_handle,
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_blocking)
        if _ !=0 : raise Exception()
        else: return obj_ang

    def get_velocity(self, obj_name):
        if obj_name in self.ddr_handles:
            obj_handle = self.ddr_handles[obj_name]
        elif obj_name in self.obj_handles:
            obj_handle = self.obj_handles[obj_name]
        elif obj_name in self.act_handles:
            obj_handle = self.act_handles[obj_name]
        else:
            return -1

        _, lin_vel_vec, ang_vel_vec = vrep.simxGetObjectVelocity(self.clientID, obj_handle,
                vrep.simx_opmode_blocking)
        if _ !=0 : raise Exception()
        else: return lin_vel_vec, ang_vel_vec

    def setJointVelocity(self, motor_names, target_velocity):
        for idx,motor_name in enumerate(motor_names):
            if motor_name in self.act_handles:
                vrep.simxSetJointTargetVelocity(self.clientID, self.act_handles[motor_name],
                            target_velocity[idx], # target velocity
                            vrep.simx_opmode_blocking)
            else:
                return -1
        vrep.simxSynchronousTrigger(self.clientID)
        return 0

    def getSimulationState(self):
        self.state = {}
        ret_list = []
        for name, handle in self.ddr_handles.items():
            lin_vel, ang_vel = self.get_velocity(name)
            x, y = self.get_position(name)[0:2]
            theta = self.get_orientation(name)[2]
            self.state[name] = [x, y, theta, np.linalg.norm(lin_vel[0:2]), ang_vel[2]]
            ret_list = self.state[name]
        for name, handle in self.obj_handles.items():
            lin_vel, ang_vel = self.get_velocity(name)
            x, y = self.get_position(name)[0:2]
            self.state[name] = [x, y, np.arctan2(lin_vel[0],lin_vel[1]) ,np.linalg.norm(lin_vel[0:2])]
            ret_list += self.state[name]


        return ret_list

    def get_handles(self, actuator_names, robot_names, object_names):
        # get the handles for each motor and set up streaming
        self.act_handles = {}
        for name in actuator_names:
            _, obj_handle = vrep.simxGetObjectHandle(self.clientID,
                    name, vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            self.act_handles.update({name:obj_handle})

        # get handle for target and set up streaming
        self.obj_handles = {}
        for name in object_names:
            _, obj_handle = vrep.simxGetObjectHandle(self.clientID,
                    name, vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            self.obj_handles.update({name:obj_handle})

        # get robot handle
        self.ddr_handles = {}
        for name in robot_names:
            _, obj_handle = vrep.simxGetObjectHandle(self.clientID,
                    name, vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            self.ddr_handles.update({name:obj_handle})

    def startSimulation(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)


