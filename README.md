# Grid World with Q-Learning (using Q-table)

### Create a new map of grid world:		
**python create_world.py**

* Provide Grid Size using Terminal.
* Define Positions of Obstacles.
* Define Positions of Agents.
* Define Positions of Goals for each agent.
* Enter name of folder in which all details of map will be stored.

[Note: To go to next step in each process, press "q"]\
You can change the size of each agent by changing value of "blocks" argument.\
You can give name to screen by "caption" argument.\
"maps" folder will contain all the folders of your maps.\
Sample examples are already provided.

### Train the agent:
**python qlearning.py --mode train**\
*lr* is learning rate\
*y* is the gamma used in Bellman Optimality Equation.\
*EPOCHS* is used to define maximum number epochs for training.\

### Test learned agent:
**python qlearning.py --mode test**\
*model* is used to define the world file used in maps/ folder.\

#### Trained Agent:
<p align="center">
  <img src="https://github.com/vinits5/grid_qlearning/blob/master/world10x10_model1/trained_agent.gif" width="256" height="256" title="Trained Agent">
</p>


#### State Values:
<p align="center">
  <img src="https://github.com/vinits5/grid_qlearning/blob/master/world10x10_model1/state_value.gif" title="State Value GIF">
</p>

Basically,
* This is part of code where you have to write functions to deal with environment.
* [World.py](https://github.com/vinits5/grid_qlearning/blob/master/World.py) file has all the defined functions.
*  It has functions similar to "openai gym" environment has.
*  "step,render,reset" are the functions to be used to train.