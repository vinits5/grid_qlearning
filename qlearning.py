import pygame
import csv
import argparse
from world_developer.world_developer import Create_World
import World
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='check', help='Modes: train/test/create_world')
parser.add_argument('--blocks', type=int, default=60, help='size of side of each square in a grid.')
parser.add_argument('--caption', type=str, default='Grid World', help='Title of screen')
parser.add_argument('--model', type=str, default='world10x10', help='Map of World')

args = parser.parse_args()
world = World.World(args.model)

LOG_PATH = '../../'+args.model+'_model1'
if args.mode == 'train':
	if not os.path.exists(LOG_PATH):
		os.mkdir(LOG_PATH)

#Declare a Q-table.
total_actions = 4
total_states = int(world.grid_size[0][0])*int(world.grid_size[0][1])
Q_table = np.zeros([total_states,total_actions])

#Parameters for Q-learning
lr = 0.8        #learning rate
y = 0.95        #gamma
EPOCHS = 201  #no of EPOCHS

def log_string(data):
	with open(os.path.join(LOG_PATH,'log'+args.mode+'.txt'),'a') as file:
		file.write(data)
		file.write('\n')

def store_data(epoch,reward,step):
	log_string('######'+str(epoch)+'######')
	log_string('Rewards: '+str(R))
	log_string('Total Steps: '+str(i))
	log_string('Average Reward: '+str(R/(i*1.0)))

if args.mode == 'train':
	
	done = [False]
	Rewards = []
	Actions = []    #No of actions taken to reach the goal.
	world.speed = 200

	for epoch in range(0,EPOCHS):
		print("######## {} ########".format(epoch))
		state,s_str = world.reset()
		world.render(60,args.caption,done)
		R = 0
		i = 0
		while i < 200:
			s_int = [int(s_str[0][0]),int(s_str[0][1])]
			table_s = world.state_table(s_int)

			# 0 --> Down
			# 1 --> Left 
			# 2 --> Right
			# 3 --> Up
			decide_action = Q_table[table_s,:]+np.random.randn(1,total_actions)*(1./(i+1))
			act = np.argmax(decide_action)
			
			s,r,done,s1_str = world.step([act])
			world.render(60,args.caption,done)

			s1_int = [int(s1_str[0][0]),int(s1_str[0][1])]
			table_s1 = world.state_table(s1_int)
			
			print('Step: {}, Initial State: {}, Reward: {}, Final State: {} & Done: {}'.format(i,s_int,r[0],s1_int,done[0]))

			Q_table[table_s,act] += lr*(r[0] + y*np.max(Q_table[table_s1,:]) - Q_table[table_s,act])

			R += r[0]
			s_str = s1_str
			i+=1

			if done[0] == True:
				print('goal')
				break
		Rewards.append(R)
		print('Reward: {}'.format(R))
		print('no goal')
		Actions.append(i)
		store_data(epoch,R,i)

		temp = epoch%10
		if temp==0:
			if not os.path.exists(os.path.join(LOG_PATH,'qtables')):
				os.mkdir(os.path.join(LOG_PATH,'qtables'))
			with open(os.path.join(LOG_PATH,'qtables','qtable'+str(epoch)+'.csv'),'w') as csvfile:
				csvwriter = csv.writer(csvfile)
				for i in range(Q_table.shape[0]):
					csvwriter.writerow(Q_table[i,:])
			print("Model Saved!")

if args.mode == 'test':
	done = [False]
	Rewards = []
	Actions = []
	world.speed = 20
	EPOCHS = 10

	Q_table = np.zeros([total_states,total_actions])

	with open(os.path.join(LOG_PATH,'qtables','qtable90.csv'),'r') as csvfile:
		csvreader = csv.reader(csvfile)
		i = 0
		for row in csvreader:
			row = np.asarray([float(x) for x in row])
			Q_table[i,:]=row
			i += 1

	for epoch in range(0,EPOCHS):
		print("######## {} ########".format(epoch))
		state,s_str = world.reset()
		
		IMG_PATH = os.path.join(LOG_PATH,'images')
		if not os.path.exists(IMG_PATH):
			os.mkdir(IMG_PATH)
		cv2.imwrite(IMG_PATH+'/image0.jpg',state)
		world.render(60,args.caption,done)
		R = 0
		i = 0
		while i < 99:
			s_int = [int(s_str[0][0]),int(s_str[0][1])]
			table_s = world.state_table(s_int)

			decide_action = Q_table[table_s,:]
			act = np.argmax(decide_action)
			
			s,r,done,s1_str = world.step([act])
			cv2.imwrite(IMG_PATH+'/image'+str(epoch)+'_model'+str(i+1)+'.jpg',s)

			world.render(60,args.caption,done)

			s1_int = [int(s1_str[0][0]),int(s1_str[0][1])]
			table_s1 = world.state_table(s1_int)
			
			print('Step: {}, Initial State: {}, Reward: {}, Final State: {} & Done: {}'.format(i,s_int,r[0],s1_int,done[0]))

			R += r[0]
			s_str = s1_str
			i+=1

			if done[0] == True:
				print('goal')
				break
		Rewards.append(R)
		Actions.append(i)
		print('Avg Reward: {}'.format(R/(i*1.0)))
		store_data(epoch,R,i)

if args.mode == 'check_efficiency':
	Efficiency = []
	for i in range(0,210,10):
		done = [False]
		Rewards = []
		Actions = []
		world.speed = 20
		EPOCHS = 2
		success = 0

		Q_table = np.zeros([total_states,total_actions])

		with open(os.path.join(LOG_PATH,'qtables','qtable'+str(i)+'.csv'),'r') as csvfile:
			csvreader = csv.reader(csvfile)
			i = 0
			for row in csvreader:
				row = np.asarray([float(x) for x in row])
				Q_table[i,:]=row
				i += 1

		for epoch in range(0,EPOCHS):
			print("######## {} ########".format(epoch))
			state,s_str = world.reset()
			
			IMG_PATH = os.path.join(LOG_PATH,'images')
			if not os.path.exists(IMG_PATH):
				os.mkdir(IMG_PATH)
			# cv2.imwrite(IMG_PATH+'/image0.jpg',state)
			world.render(60,args.caption,done)
			R = 0
			i = 0
			while i < 1:
				s_int = [int(s_str[0][0]),int(s_str[0][1])]
				table_s = world.state_table(s_int)

				decide_action = Q_table[table_s,:]
				act = np.argmax(decide_action)
				
				s,r,done,s1_str = world.step([act])
				# cv2.imwrite(IMG_PATH+'/image'+str(epoch)+'_model'+str(i+1)+'.jpg',s)

				world.render(60,args.caption,done)

				s1_int = [int(s1_str[0][0]),int(s1_str[0][1])]
				table_s1 = world.state_table(s1_int)
				
				print('Step: {}, Initial State: {}, Reward: {}, Final State: {} & Done: {}'.format(i,s_int,r[0],s1_int,done[0]))

				R += r[0]
				s_str = s1_str
				i+=1

				if done[0] == True:
					print('goal')
					success += 1
					break
			Rewards.append(R)
			Actions.append(i)
			print('Avg Reward: {}'.format(R/(i*1.0)))
			store_data(epoch,R,i)
		print(success)
		Efficiency.append(success/100.0)
		print(Efficiency)	
	with open('../../efficiency.csv','w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(Efficiency)	


if args.mode == 'check':
	print("Please provide mode to run!!")
	print("USE:")
	print("python qlearning.py --mode train")
	print("OR")
	print("python qlearning.py --mode test")