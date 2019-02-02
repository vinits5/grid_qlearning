import pygame
import argparse
import math
import cv2
import csv
import os
import random

pygame.init()

class Create_World:
	def create_world(self,grid_width,grid_height,caption,blocks):
		# grid_blocks_width, grid_blocks_height are the number of blocks in grid.
		self.grid_blocks_width = grid_width
		self.grid_blocks_height = grid_height

		self.blocks = blocks

		# This is the actual screen size.
		self.grid_height = grid_height*self.blocks
		self.grid_width = grid_width*self.blocks

		self.black = (0,0,0)
		self.white = (255,255,255)
		self.red = (255,0,0)
		self.green = (0,255,0)
		self.blue = (0,0,255)
		self.gray = (89,89,89)
		self.agent1 = (51,51,204)
		
		self.gameDisplay = pygame.display.set_mode((self.grid_width,self.grid_height))
		self.gameDisplay.fill(self.white)
		pygame.display.set_caption(caption)
		self.clock = pygame.time.Clock()

	def gridPos_realPos(self,pos):
		return [pos[0]*self.blocks,pos[1]*self.blocks]

	# The top-left corner is the reference point for everything.
	def obstacles(self,pos):
		# Draw obstacles
		pos = self.gridPos_realPos(pos)
		pygame.draw.rect(self.gameDisplay,self.gray,[pos[0],pos[1],self.blocks,self.blocks])

	def agent(self,pos,color):
		# Draw agents
		pos = self.gridPos_realPos(pos)
		pygame.draw.circle(self.gameDisplay,color,[pos[0]+(self.blocks/2),pos[1]+(self.blocks/2)],self.blocks/2)

	def goal(self,pos,color):
		# Draw goal
		pos = self.gridPos_realPos(pos)
		pygame.draw.circle(self.gameDisplay,color,[pos[0]+(self.blocks/2),pos[1]+(self.blocks/2)],self.blocks/2,2)

	def mouse_to_gridPos(self,pos):
		_,pos[0] = math.modf((pos[0]/self.blocks))
		_,pos[1] = math.modf((pos[1]/self.blocks))
		pos = [int(pos[0]),int(pos[1])]
		return pos

	def color_defination(self):
		r = random.randint(0,255)
		g = random.randint(0,255)
		b = random.randint(0,255)
		return (r,g,b)

	def create_obstacles(self,done):
		self.obstacles_poses = []
		store_poses = False
		while not done:	
			for event in pygame.event.get():
				if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					# x -> along width and y -> height
					x,y = event.pos
					pos = self.mouse_to_gridPos([x,y])
					self.obstacles_poses.append(pos)
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						store_poses = True
						done = True
			# print(self.obstacles_poses)
			if len(self.obstacles_poses)>0:
				for i in range(len(self.obstacles_poses)):
					self.obstacles(self.obstacles_poses[i])
			pygame.display.update()
			self.clock.tick(60)
		return done

	def check_if_obstacles(self,pos):
		checked = True
		for i in range(len(self.obstacles_poses)):
			if (self.obstacles_poses[i][0]==pos[0] and self.obstacles_poses[i][1] == pos[1]):
				checked = False
		return checked

	def check_if_agents(self,pos):
		checked = True
		for i in range(len(self.agents_poses)):
			if (self.agents_poses[i][0]==pos[0] and self.agents_poses[i][1] == pos[1]):
				checked = False
		return checked

	def create_agents(self,done):
		self.agents_poses = []
		self.agents_color = []
		while not done:	
			for event in pygame.event.get():
				if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					# x -> along width and y -> height
					x,y = event.pos
					pos = self.mouse_to_gridPos([x,y])
					checked = self.check_if_obstacles(pos)
					self.agents_color.append(self.color_defination())
					if checked:
						self.agents_poses.append(pos)
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						done = True
			if len(self.agents_poses)>0:
				for i in range(len(self.agents_poses)):
					self.agent(self.agents_poses[i],self.agents_color[i])
			pygame.display.update()
			self.clock.tick(60)
		return done

	def create_goals(self,done):
		self.goal_poses = []
		while not done:	
			for event in pygame.event.get():
				if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					# x -> along width and y -> height
					x,y = event.pos
					pos = self.mouse_to_gridPos([x,y])
					checked_o = self.check_if_obstacles(pos)
					checked_a = self.check_if_agents(pos)
					if checked_o and checked_a:
						self.goal_poses.append(pos)
				if len(self.goal_poses) == len(self.agents_poses):
					done = True
			# print(self.obstacles_poses)
			if len(self.goal_poses)>0:
				for i in range(len(self.goal_poses)):
					self.goal(self.goal_poses[i],self.agents_color[i])
			pygame.display.update()
			self.clock.tick(60)
		return done
	
	def image_state(self):
		rgb = pygame.surfarray.pixels3d(pygame.transform.rotate(self.gameDisplay,270))
		rgb = cv2.flip(rgb,0)
		return rgb

	def store_map(self):
		folder = input('Enter name of folder: ')
		path = os.getcwd()
		path = os.path.join(path,'maps')
		os.chdir(path)
		os.mkdir(folder)
		path = os.path.join(path,folder)
		os.chdir(path)
		with open('grid.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([self.grid_blocks_width,self.grid_blocks_height])
		with open('obstacle.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			for i in range(len(self.obstacles_poses)):
				csvwriter.writerow(self.obstacles_poses[i])
		with open('agent.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			for i in range(len(self.agents_poses)):
				csvwriter.writerow(self.agents_poses[i])
		with open('goal.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			for i in range(len(self.goal_poses)):
				csvwriter.writerow(self.goal_poses[i])
		with open('color.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			for i in range(len(self.agents_color)):
				csvwriter.writerow(self.agents_color[i])