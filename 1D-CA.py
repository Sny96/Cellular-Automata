import numpy as np
import random

class Environment():
   
   r_x=0.4
   r_y=0.3
   max_move = 5
   array_length = 100
   
   def init(self):
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
      reproduce() #spawn 
      update_health() #increase prey health by r_x, reduce pred health by r_y

   
   def get_preylist(self):
      return self.prey_list

   def get_predlist(self):
      return self.pred_list

   def get_numPreyPred():
      return sum(self.prey_list), sum(self.pred_list) #Not true, needs to change
   
   def move_pred(): #How should we move our predators?? random?
      print(1)
      
      
   def update_health():
      self.pred_list = [(x-self.r_y) for x in self.pred_list if x!=0]
      self.pred_list = [0 for x in self.pred_list if x<0]
   
      self.prey_list = [(x+self.r_x) for x in self.prey_list if x!=0]
   
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
   