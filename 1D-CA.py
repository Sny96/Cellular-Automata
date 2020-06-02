import numpy as np
import random
from parameters import *
import matplotlib.pyplot as plt

class Environment():
   
   def init(self):
      self.seed = seed
      random.seed(self.seed)
      self.eps = 1E-3
      
      self.r_x = r_x
      self.r_y = r_y
      self.max_move = max_move
      self.array_length = array_length
      
      self.prey_list = [0]*self.array_length
      self.pred_list = [0]*self.array_length
      
      self.time=0
      
   def init_pred_prey(self, n_prey, n_pred):
      """creates the initial population of preys/predators"""
      #obtain indexing
      index_pred = [index for index, value in enumerate(self.pred_list)]
      index_prey = [index for index, value in enumerate(self.prey_list)]
      
      #select locations that will contain initial preys/predators
      r_pred = random.choices(index_pred, k=n_pred)
      r_prey = random.choices(index_prey, k=n_prey)
      #populate the lists
      for i in r_pred:
         self.pred_list[i] = 1
      for i in r_prey:
         self.prey_list[i] = 1
   
   def get_preylist(self):
      return self.prey_list

   def get_predlist(self):
      return self.pred_list
   
   def get_creatures(self):
      return sum(self.prey_list), sum(self.pred_list)
   
   def move_all_pred(self): #chosen at random, moved in random direction
      pred_index_list = [index for index, value in enumerate(self.pred_list) if value == 1]
      
      for i in range(len(pred_index_list)):
         
         pred_index_list = [index for index, value in enumerate(self.pred_list) if value == 1]
         r = random.choice(pred_index_list)
         direction = random.choice([-1,1])
         dist = self.pred_list[(r+direction)%self.array_length]-self.pred_list[r]
         move = max(self.max_move, abs(dist)-1)
         self.pred_list[r]=0
         self.pred_list[(r+move*direction)%self.array_length] = 1
      
      
   def update_rep_death(self):
      index_pred = [index for index, value in enumerate(self.pred_list)]
      index_prey = [index for index, value in enumerate(self.prey_list)]
      random.shuffle(index_prey)
      random.shuffle(index_pred)
            
      for i in index_prey:
         if random.random()<self.r_x:
            self.spawn_new_prey(i)
            
      for i in index_prey:
         if random.random()<self.r_y:
            self.pred_list[i] = 0
            
            
   def spawn_new_prey(self, i): 
      if self.prey_list[i] == 0:
         self.prey_list[i] = 1
      else:
         r = random.choice([-1,1])
         if self.prey_list[(i+r)%self.array_length]==0:
            self.prey_list[(i+r)%self.array_length] = 1
         elif self.prey_list[(i-r)%self.array_length]==0:
            self.prey_list[(i-r)%self.array_length] = 1
   
   def spawn_new_pred(self, i): 
      r = random.choice([-1,1])
      if self.pred_list[(i+r)%self.array_length]==0:
         self.pred_list[(i+r)%self.array_length] = 1
      elif self.pred_list[(i-r)%self.array_length]==0:
         self.pred_list[(i-r)%self.array_length] = 1
            
   def interaction(self):
      index_list = []
      for i in range(self.array_length):
         if (self.prey_list[i] == 1 and self.pred_list[i] == 1):
            index_list.append(i)
      random.shuffle(index_list)
      for index in index_list:   
            self.prey_list[index] = 0
            self.spawn_new_pred(index)

         
   def update(self):
      self.time+=1
      self.move_all_pred() #every pred moves
      self.interaction() #pred health->1, prey->0 when interacting 
      self.update_rep_death() #increase prey health by r_x, reduce pred health by r_y

def plotting_system(pop_matrix, name):
      fig = plt.figure(num=None, figsize=(6, 13), dpi=80, facecolor='w', edgecolor='k')
      plt.xlabel('Space')
      plt.ylabel('Time')
      plt.title(name)
      plt.imshow(pop_matrix, cmap='Greys', origin='lower')
      
def plotting_population(num_pred_list, num_prey_list):
    timesteps = list(range(time_of_sim + 1))    #used to color-code the relative time
    plt.xlabel('Predators')
    plt.ylabel('Prey')
    plt.title("Preys Vs Predators for all Time-Steps")
    #c=timesteps colorcodes the scatterplot so that earlier datapoint are violet and newer are yellow
    plt.scatter(num_pred_list, num_prey_list, c=timesteps)

def main():
   a = Environment()        #create instance of the class
   a.init()
   a.init_pred_prey(prey_num,pred_num)  #create the initial placements of predators/preys
   predator_list = [[x for x in a.get_predlist()]]
   prey_list = [[x for x in a.get_preylist()]]
   
   c = a.get_creatures()
   num_prey = [c[0]]    #stores the number of preys at each time-step
   num_pred = [c[1]]    #stores the number of predators at each time-step
   
   #runs the simulation though time-steps
   for i in range(time_of_sim):
      a.update()

      prey_list.append([x for x in a.get_preylist()])
      predator_list.append([x for x in a.get_predlist()])
      
      c = a.get_creatures()
      num_prey.append(c[0])
      num_pred.append(c[1])
      
   #diplay the data
   plotting_population(num_pred, num_prey)
   plotting_system(prey_list, 'Preys')
   plotting_system(predator_list, 'Predators')

if __name__ == "__main__":
   main()
      
   