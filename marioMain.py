#creating environment to play super mario#
#import game
import gym_super_mario_bros
#import joypad wrapper
from nes_py.wrappers import JoypadSpace
#import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#import frame stacker and grey scale wrappers
from gym.wrappers import GrayScaleObservation
#import vectorisation wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#import matplotlib
from matplotlib import pyplot as plt

def randomActionTest():
    #setup game
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env,SIMPLE_MOVEMENT)
    #create loop of random actions to take
    #cretae a flag - restart game or not
    done = True
    #loop through game frames
    for step in range(100000):
        if done:
            #start the game
            env.reset()
        #perform an action in game for this frame (at the moment steps are random)
        state, reward, done, info = env.step(env.action_space.sample())
        #show game on screen
        env.render()
    #close game
    env.close()
    return True

def preProcess(env):
    #1. create base environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #2. simplify controls
    env = JoypadSpace(env,SIMPLE_MOVEMENT)
    #3. greyscale
    env = GrayScaleObservation(env, keep_dim=True)
    #4. wrap in dummy environment
    env = DummyVecEnv([lambda:env])
    #5 stack the frames (4 is variable parameter but 4 works)
    env = VecFrameStack(env,4,channels_order='last')

    return env


if __name__ == '__main__':
