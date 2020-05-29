import numpy as np
import random

class Environment():
   
   def init(self):
      def __init__(self, array_length):
        self.array_length = array_length
        self.prey_list = [0]*self.array_length
        self.predators_list = [0]*self.array_length
        
        self.max_move = 5
        self.time=0
   
      def update():
         self.time+=1
         #TODO: update how prey and predators move
         
      def get_preylist():
         return self.prey_list
      
      def get_predlist():
         return self.pred_list
      
      def get_numPreyPred():
         return sum(self.prey_list), sum(self.pred_list)
         
      def change_prey(index,num):
         
         index_list = [index for index, value in enumerate(self.prey_list) if value == 0]
         #TODO: decide initialization rule
         if num==0:
            self.prey_list[index] = 0
         if num==1:
            self.prey_list[index] = 1
         
         
      