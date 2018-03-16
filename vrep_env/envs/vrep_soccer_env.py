#import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import vrep
import os
import time, random
import numpy as np
from operator import sub

class VrepSoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ip = '127.0.0.1'
        port = 19997
        time_step = 0.01
        actuator_names = ['LeftMotor', 'RightMotor']
        object_names = ['Bola']
        dummy_names = ['linha_fundo_baixo', 'linha_fundo2_alto']
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

        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)

        self.get_handles(actuator_names, robot_names, object_names, dummy_names)

        self.time_step = time_step
        #todo check if observation_space and action_space are correct
        shape = len(robot_names)*5 + len(object_names)*4
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32, shape=(shape,))
        shape = len(robot_names)*2
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(shape,))

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
        angles_diff = list(map(sub, angles, self.init_orientation))
        if (abs(angles_diff[0]) >10 or abs(angles_diff[1]) > 10):
            print("*******Robo virou*****")
            return True

        ball_pos = np.array(self.get_position('Bola'))
        if (abs(ball_pos[2]) > 10):
            print("*******Bola caiu no Limbo*****")
            return True

        ball_pos = ball_pos[0:2]
        robot_pos = np.array(self.state['DifferentialDriveRobot'][0:2])
        if (np.linalg.norm(robot_pos - ball_pos) < 0.1):
            print("*******Objetivo Alcancado*****")
            return True

        return False

    def _take_action(self, action):
        if (abs(action[0]) > 1):
            l = 1 if action[0] > 0 else -1
        else:
            l = action[0]

        if (abs(action[1]) > 1):
            omega = 1 if action[1] > 0 else -1
        else:
            omega = action[0]
        l = l/10
        HALF_AXIS = 0.0325
        WHEEL_R = 0.032/2
        motorSpeed = [0,0]
        #transfer input to motor velocities
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
        robot_pos = np.array(self.state['DifferentialDriveRobot'][0:2])
        target_pos = np.array(self.state['Bola'][0:2])
        time1_fundo_pos = np.array(self.state['linha_fundo_baixo'])
        time2_fundo_pos = np.array(self.state['linha_fundo2_alto'])

        #check gol "a favor" (lado negativo do campo)
        #if (target_pos[0] < time2_fundo_pos[0])

        #check got "contra" (lado positivo do campo)
        #if (target_pos[0] > time2_fundo_pos[0])

        #reward to robot to follow ball position
        reward = 1/np.linalg.norm(robot_pos - target_pos)

        return reward

    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)

        time1_fundo_pos = np.array(self.state['linha_fundo_baixo'])
        time2_fundo_pos = np.array(self.state['linha_fundo2_alto'])

        for ddr_name, ddr_handle in self.ddr_handles.items():
            new_x = random.uniform(time2_fundo_pos[0] + 0.2, time1_fundo_pos[0] - 0.2)
            new_y = random.uniform(time2_fundo_pos[1] + 0.2, time1_fundo_pos[1] - 0.2)
            vrep.simxSetObjectPosition(self.clientID, ddr_handle, -1, 
                                        [new_x, new_y, self.get_position(ddr_name)[2]], #new position
                                        vrep.simx_opmode_blocking)

        for obj_name, obj_handle in self.obj_handles.items():
            new_x = random.uniform(time2_fundo_pos[0] + 0.2, time1_fundo_pos[0] - 0.2)
            new_y = random.uniform(time2_fundo_pos[1] + 0.2, time1_fundo_pos[1] - 0.2)
            vrep.simxSetObjectPosition(self.clientID, obj_handle, -1, 
                                        [new_x, new_y, self.get_position(obj_name)[2]], #new position
                                        vrep.simx_opmode_blocking)

        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(.05)
        self._turn_display(False)

        print("RESET")

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
        elif obj_name in self.dummy_handles:
            obj_handle = self.dummy_handles[obj_name]
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

        for name, handle in self.dummy_handles.items():
            x, y = self.get_position(name)[0:2]
            self.state[name] = [x, y] 

        return np.transpose(np.array(ret_list))

    def get_handles(self, actuator_names, robot_names, object_names, dummy_names):
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
        # get dummy handles
        self.dummy_handles = {}
        for name in dummy_names:
            _, obj_handle = vrep.simxGetObjectHandle(self.clientID,
                    name, vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            self.dummy_handles.update({name:obj_handle})

    def startSimulation(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)


