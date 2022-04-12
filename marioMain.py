#creating environment to play super mario#
#import game
import gym_super_mario_bros
#import joypad wrapper
from nes_py.wrappers import JoypadSpace
#import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def randomActionTest(env):
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

if __name__ == '__main__':
    #setup game
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env,SIMPLE_MOVEMENT)

    randomActionTest(env)
