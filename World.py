# Class for grid world RL algorithms.
# All functions are exactly similar to openai gym.
# Pre-requisite for grid world game is numpy, opnecv and pygame.

import pygame
import random
import os
import csv
from copy import copy
import cv2
from IPython import embed

class World:
	# Initialization of all datas from the given map/grid folder.
	def __init__(self,name):
		path = os.getcwd()
		path = os.path.join(path,'maps',name)
		os.chdir(path)
		# Lists for obstacle,agent,goal & colors are defined.
		# List "agent_steps" is used to track the agent position.
		self.obstacle_poses,self.agent_poses,self.agent_colors,self.goal_poses,self.grid_size,self.agent_steps = [],[],[],[],[],[]
		with open('obstacle.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				self.obstacle_poses.append(row)
		with open('agent.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				self.agent_poses.append(row)
		with open('color.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				self.agent_colors.append(row)
		with open('grid.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				self.grid_size.append(row)
		with open('goal.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				self.goal_poses.append(row)
		self.agent_steps = copy(self.agent_poses)
		self.speed = 20

	# Basic Colors are defined here.
	def colors(self):
		self.black = (0,0,0)
		self.white = (255,255,255)
		self.red = (255,0,0)
		self.green = (0,255,0)
		self.blue = (0,0,255)
		self.gray = (89,89,89)
		self.agent1 = (51,51,204)

	# Conver grid position to pixel position.
	def gridPos_realPos(self,pos):
		return [pos[0]*self.blocks,pos[1]*self.blocks]

	# The top-left corner is the reference point for everything.
	# To draw an obstacle at given position on display.
	def obstacles(self,pos):
		# Draw obstacles
		pos = self.gridPos_realPos(pos)
		pygame.draw.rect(self.gameDisplay,self.gray,[pos[0],pos[1],self.blocks,self.blocks])

	# To draw an agent at given position on display with given color.
	def agent(self,pos,color):
		# Draw agents
		pos = self.gridPos_realPos(pos)
		# Draw circle as agent.
		pygame.draw.circle(self.gameDisplay,color,[pos[0]+(self.blocks/2),pos[1]+(self.blocks/2)],self.blocks/2)
		# Draw rectangle as agent.
		# pygame.draw.rect(self.gameDisplay,color,[pos[0],pos[1],self.blocks,self.blocks])
	# To draw a goal at given position on display with given color.
	def goal(self,pos,color):
		# Draw goal
		pos = self.gridPos_realPos(pos)
		# Draw circle as goal.
		# pygame.draw.circle(self.gameDisplay,color,[pos[0]+(self.blocks/2),pos[1]+(self.blocks/2)],self.blocks/2,2)

		# Draw filled circle as goal.
		# pygame.draw.circle(self.gameDisplay,color,[pos[0]+(self.blocks/2),pos[1]+(self.blocks/2)],self.blocks/2)

		# Draw plus sign as goal.
		pygame.draw.rect(self.gameDisplay,color,[pos[0]+(self.blocks-10)/2,pos[1]+10,10,self.blocks-20])
		pygame.draw.rect(self.gameDisplay,color,[pos[0]+10,pos[1]+(self.blocks-10)/2,self.blocks-20,10])

	# Render the environment to visualize the grid.
	def render(self,blocks,caption,done):
		self.colors()
		self.blocks = blocks
		# Create pygame display.
		self.gameDisplay = pygame.display.set_mode((int(self.grid_size[0][0])*blocks,int(self.grid_size[0][1])*blocks))
		# Define clock.
		self.clock = pygame.time.Clock()
		# Set caption for the display.
		pygame.display.set_caption(caption)
		# Fill the display with white color.
		self.gameDisplay.fill(self.white)
		
		# Draw all obstacles on the display.
		for i in range(len(self.obstacle_poses)):
			self.obstacles([int(self.obstacle_poses[i][0]),int(self.obstacle_poses[i][1])])
		# Draw all agents on the display.
		for i in range(len(self.agent_steps)):
			self.agent([int(self.agent_steps[i][0]),int(self.agent_steps[i][1])],(int(self.agent_colors[i][0]),int(self.agent_colors[i][1]),int(self.agent_colors[i][2])))
		# Draw all goals on the display.
		for i in range(len(self.goal_poses)):
			color = (int(self.agent_colors[i][0]),int(self.agent_colors[i][1]),int(self.agent_colors[i][2]))
			# Check agent state.
			if done[i] == True:
				# If agent is at goal. Change color of goal sign to black.
				color = (0,0,0)
			self.goal([int(self.goal_poses[i][0]),int(self.goal_poses[i][1])],color)	

		for i in range(0,5):
			pygame.display.update()
			# self.clock.tick(20)
			self.clock.tick(self.speed)

	# Check if the agent collides with obstacle after step has taken.
	def check_obstacle(self,pos):
		# "pos" is the state of agent after step has taken.
		collided = False
		for i in range(len(self.obstacle_poses)):
			if pos == self.obstacle_poses[i]:
				collided = True
		# True if pos state has obstacle else False.
		return collided

	# Check if the agent is going out of grid.
	def check_limits(self,pos):
		# "pos" is state of agent after step has taken.
		out_of_grid = False
		if int(pos[0])<0 or int(pos[1])<0 or int(pos[0]) == int(self.grid_size[0][0]) or int(pos[1]) == int(self.grid_size[0][1]):
			out_of_grid = True
		# True if pos state is out of grid else False.
		return out_of_grid

	# Check if agent has reached goal state.
	def check_goal(self,pos,i):
		# "pos" is state of agent after step has taken.
		# "i" is the goal element for the list of goals.
		on_goal = False
		if pos == self.goal_poses[i]:
			on_goal = True
		# True if pos state is goal state for that agent else False.
		return on_goal

	# Reward achieved by the agent after step has taken.
	def reward_function(self,collided,pos,i,out_of_grid,on_goal):
		# "pos" is the new state of agent before taking any step.
		# Check if agent is already on goal (IMP: in case of multiple agents.)
		if not on_goal:
			on_goal = self.check_goal(pos,i)
		# If agent hits the obstacle.
		if collided:
			reward = -3
		# If agent goes out of grid.
		elif out_of_grid:
			reward = -3
		# If agent reaches goal.
		elif on_goal:
			reward = 10
		# If agent takes a step.
		else:
			reward = 1
		return reward

	# Environment state as an image.
	def image_state(self,blocks,done):
		# Define colors
		self.colors()
		self.blocks = blocks
		# Define pygame display.
		self.gameDisplay = pygame.display.set_mode((int(self.grid_size[0][0])*blocks,int(self.grid_size[0][1])*blocks))
		self.gameDisplay.fill(self.white)
		
		# Draw obstacles.
		for i in range(len(self.obstacle_poses)):
			self.obstacles([int(self.obstacle_poses[i][0]),int(self.obstacle_poses[i][1])])
		# Draw agents as per current states.
		for i in range(len(self.agent_steps)):
			self.agent([int(self.agent_steps[i][0]),int(self.agent_steps[i][1])],(int(self.agent_colors[i][0]),int(self.agent_colors[i][1]),int(self.agent_colors[i][2])))

		# Draw goals.
		for i in range(len(self.goal_poses)):
			color = (int(self.agent_colors[i][0]),int(self.agent_colors[i][1]),int(self.agent_colors[i][2]))
			if done[i] == True:
				# Change goal color to black if agent is in goal state.
				color = (0,0,0)
			self.goal([int(self.goal_poses[i][0]),int(self.goal_poses[i][1])],color)	

		# Convert display to rgb numpy array.
		rgb = pygame.surfarray.array3d(pygame.transform.rotate(self.gameDisplay,270))
		# rgb = pygame.surfarray.array3d(self.gameDisplay)
		rgb = cv2.flip(rgb,0)
		for i in range(rgb.shape[0]):
			for j in range(rgb.shape[1]):
				temp = rgb[i][j][2]
				rgb[i][j][2] = rgb[i][j][0]
				rgb[i][j][0] = temp
		return rgb

	# Take step for agent to get next_state,reward,done.
	def step(self,action):
		state,rewards,done,agent_pos = [],[],[],[]
		# Check if actions for all agents are there.
		if len(action)==len(self.agent_steps):
			# Take step for each agent.
			for i in range(len(self.agent_steps)):
				# pos is the list of new states for each agent.
				# pos_i is the list of initial states for each agent.
				pos = []
				pos_i = []

				# 0 --> Down
				# 1 --> Left 
				# 2 --> Right
				# 3 --> Up

				if action[i] == 0:
					# Update pos and pos_i lists.
					pos_i.append(str(int(self.agent_steps[i][0])))
					pos_i.append(str(int(self.agent_steps[i][1])))
					pos.append(str(int(self.agent_steps[i][0])))
					pos.append(str(int(self.agent_steps[i][1])+1))

					# Check if agent hits obstacle. So, there won't be any change in new state.
					collided = self.check_obstacle(pos)
					# Check if agent goes out of grid. So, there won't be any change in new state.
					out_of_grid = self.check_limits(pos)
					# Check if agent is initially on goal. So, there won't be any change in new state.
					on_goal = self.check_goal(pos_i,i)
					# Find reward for the transition.
					reward = self.reward_function(collided,pos,i,out_of_grid,on_goal)
					# print(collided,out_of_grid,on_goal)

					# Update the agent_steps list if agent hasn't collided, not gone out of grid or reached goal.
					if not (collided or out_of_grid or on_goal):
						self.agent_steps[i] = pos
					# Check if the agent has reached goal.
					if not on_goal:
						on_goal = self.check_goal(pos,i)

				# The algorithm for all other actions is exactly same as of action "0".
				if action[i] == 1:
					pos_i.append(str(int(self.agent_steps[i][0])))
					pos_i.append(str(int(self.agent_steps[i][1])))
					pos.append(str(int(self.agent_steps[i][0])-1))
					pos.append(str(int(self.agent_steps[i][1])))
					collided = self.check_obstacle(pos)
					out_of_grid = self.check_limits(pos)
					on_goal = self.check_goal(pos_i,i)
					reward = self.reward_function(collided,pos,i,out_of_grid,on_goal)
					# print(collided,out_of_grid,on_goal)
					if not (collided or out_of_grid or on_goal):
						self.agent_steps[i] = pos
					if not on_goal:
						on_goal = self.check_goal(pos,i)
				if action[i] == 2:
					pos_i.append(str(int(self.agent_steps[i][0])))
					pos_i.append(str(int(self.agent_steps[i][1])))
					pos.append(str(int(self.agent_steps[i][0])+1))
					pos.append(str(int(self.agent_steps[i][1])))
					collided = self.check_obstacle(pos)
					out_of_grid = self.check_limits(pos)
					on_goal = self.check_goal(pos_i,i)
					reward = self.reward_function(collided,pos,i,out_of_grid,on_goal)
					# print(collided,out_of_grid,on_goal)
					if not (collided or out_of_grid or on_goal):
						self.agent_steps[i] = pos
					if not on_goal:
						on_goal = self.check_goal(pos,i)
				if action[i] == 3:
					pos_i.append(str(int(self.agent_steps[i][0])))
					pos_i.append(str(int(self.agent_steps[i][1])))
					pos.append(str(int(self.agent_steps[i][0])))
					pos.append(str(int(self.agent_steps[i][1])-1))
					collided = self.check_obstacle(pos)
					out_of_grid = self.check_limits(pos)
					on_goal = self.check_goal(pos_i,i)
					reward = self.reward_function(collided,pos,i,out_of_grid,on_goal)
					# print(collided,out_of_grid,on_goal)
					if not (collided or out_of_grid or on_goal):
						self.agent_steps[i] = pos
					if not on_goal:
						on_goal = self.check_goal(pos,i)
				agent_pos.append(self.agent_steps[i])
				rewards.append(reward)
				done.append(on_goal)
		state = self.image_state(60,done)
		return state,rewards,done,agent_pos

	def random_pos(self,width,height):
		result = False
		while not result:
			pos = [random.randint(0,width-1),random.randint(0,height-1)]
			result = self.check_pos(pos)
		return pos

	def check_pos(self,pos):
		result = True
		for i in range(len(self.obstacle_poses)):
			if (pos[0]==int(self.obstacle_poses[i][0]) and pos[1]==int(self.obstacle_poses[i][1])):
				result = False
		for i in range(len(self.goal_poses)):
			if (pos[0]==int(self.goal_poses[i][0]) and pos[1]==int(self.goal_poses[i][1])):
				result = False
		return result

	# Reset the environment to initial conditions for agent states.
	def reset(self):
		# Update agent poses.
		for i in range(len(self.agent_poses)):
			pos = self.random_pos(int(self.grid_size[0][0]),int(self.grid_size[0][1]))
			self.agent_poses[i] = [str(pos[0]),str(pos[1])]
		# print(self.agent_poses)
		self.agent_steps = copy(self.agent_poses)
		# print self.agent_steps
		done = []
		for i in range(len(self.agent_steps)):
			done.append(False)
		# Take initial state.
		state = self.image_state(60,done)
		return state,self.agent_steps

	def state_table(self,state):
		#row 1: 0 1 2  3
		#row 2: 4 5 6  7
		#tow 3: 8 9 10 11
		# print('Grid Size: {}'.format(self.grid_size[0][1]))
		# print(state[0]+state[1]*int(self.grid_size[0][1]))
		return state[0]+state[1]*int(self.grid_size[0][1])