import numpy as np
import random

class Environment():
   
   def init(self):
      self.seed = 1
      random.seed(self.seed)
      self.eps = 1E-3
      
      self.r_x = 0.4
      self.r_y = 0.3
      self.max_move = 5
      self.array_length = 100
      
      self.prey_list = [0]*self.array_length
      self.pred_list = [0]*self.array_length
      self.old_pred_list = [0]*self.array_length
      
      self.time=0
   
   
   def update():
      self.time+=1
      self.old_pred_list = [x for x in pred_list]
      
      move_pred() #every pred moves
      interaction() #pred health->1, prey->0 when interacting 
      update_rep_death() #increase prey health by r_x, reduce pred health by r_y

   
   def get_preylist(self):
      return self.prey_list

   def get_predlist(self):
      return self.pred_list
   
   def move_pred(): #How should we move our predators?? random?
      print(1)
      
      
   def update_rep_death(self):
      
      for i in range(self.array_length):
         if random.random()<self.r_x and self.prey_list[i]==1:
            spawn_new_prey(i)
         if random.random()<self.r_y and self.pred_list[i]==1:
            self.pred_list[i] = 0
      
   def interaction(self):
      index_list = []
      for i in range(self.array_length):
         if (self.prey_list[i] != 0 and self.pred_list != 0):
            index_list.append(i)
      
      for index in random.choice(index_list):
            self.prey_list[index] = 0
            spawn_new_pred(index)
            
   def spawn_new_prey(self, i): 
      if self.prey_list[i] == 0:
         self.prey_list = 1
      else:
         r = random.choice([-1,1])
         if self.prey_list[(i+r)%self.array_length]==0:
            self.prey_list[(i+r)%self.array_length] = 1
         elif self.prey_list[(i-r)%self.array_length]==0:
            self.prey_list[(i-r)%self.array_length] = 1
   
   def spawn_new_pred(self, i): 
      if self.pred_list[i] == 0:
         self.pred_list = 1
      else:
         r = random.choice([-1,1])
         if self.pred_list[(i+r)%self.array_length]==0:
            self.pred_list[(i+r)%self.array_length] = 1
         elif self.pred_list[(i-r)%self.array_length]==0:
            self.pred_list[(i-r)%self.array_length] = 1
   
   def change_prey(index,num):
      
      index_list = [index for index, value in enumerate(self.prey_list) if value == 0]
      #TODO: decide initialization rule
      if num==0:
         self.prey_list[index] = 0
      if num==1:
         self.prey_list[index] = 1


if __name__ == "__main__":
   a = Environment()
   a.init()
   b = a.get_preylist()
   print(b)
   