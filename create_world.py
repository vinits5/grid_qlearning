import pygame
import csv
import argparse
from world_developer.world_developer import Create_World
import World
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--blocks', type=int, default=60, help='size of side of each square in a grid.')
parser.add_argument('--caption', type=str, default='Grid World', help='Title of screen')

args = parser.parse_args()

w = input('Enter Grid Width: ')
h = input('Enter Grid Height: ')
world = Create_World()
world.create_world(w,h,args.caption,args.blocks)
done,agent_done,goal_done = False,False,False
print('\nDefine obstacles and press "q" once you are done!')
done = world.create_obstacles(done)
print('\nDefine agents and press "q" once you are done!')
agent_done = world.create_agents(agent_done)
print('\nDefine goal location for all agents!')
goal_done = world.create_goals(goal_done)
pygame.display.update()
world.clock.tick(60)

print('\nEnter name of folder to store the map (string fromat)')
world.store_map()
pygame.quit()
