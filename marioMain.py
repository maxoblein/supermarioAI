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
#import os for file management
import os
#import PPO to get the algo
from stable_baselines3 import PPO
#import base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path,exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


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

def preProcess():
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

def AIPlayMario(env,model):
    #start game
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()

def trainMarioModel():
    #get preprocessed environment
    env = preProcess()

    CheckpointDir = 'train/'
    LogDir = 'logs/'

    #set up model saving callback
    callback = TrainAndLoggingCallback(check_freq = 10000, save_path = CheckpointDir)

    #set up model
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LogDir, learning_rate = 0.0001, n_steps = 512)

    #train the model
    model.learn(total_timesteps = 500000, callback = callback)

if __name__ == '__main__':


    #test the model at playing supermario
    env = preProcess()
    #train if needed
    #trainMarioModel()
    #load model
    model = PPO.load('train/best_model_200000')

    AIPlayMario(env,model)
