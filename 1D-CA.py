import numpy as np
import random
from parameters import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools
import os

class Environment():

   def init(self, seed_):
      self.seed = seed_
      random.seed(self.seed)
      self.eps = 1E-3

      self.r_x = r_x
      self.r_y = r_y
      self.max_move = max_move
      self.array_length = array_length

      #the lists that represent the system at each time-step
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

   def update_rep_death(self):
      """Based on the rates of reproduction (r_x) and starvation (r_y), updates the populations."""
      index_pred = [index for index, value in enumerate(self.pred_list) if value==1]
      index_prey = [index for index, value in enumerate(self.prey_list) if value==1]
      #rearrages the order to remove bias
      random.shuffle(index_prey)
      random.shuffle(index_pred)
      #spawns new prey
      for i in index_prey:
         if random.random()<self.r_x:
            self.spawn_new_prey(i)
      #kills predators
      for i in index_pred: #potential bug fix
         if random.random()<self.r_y:
            self.pred_list[i] = 0
         
   def move_all_pred(self): #chosen at random, moved in random direction
      
      n = self.get_creatures()[1]
      for i in range(n):
         pred_index_list = [index for index, value in enumerate(self.pred_list) if value == 1]
         r = random.choice(pred_index_list)
         r = pred_index_list.index(r)
         direction = random.choice([-1,1])
         if 0<=r+direction<=len(pred_index_list)-1:
            dist = abs(pred_index_list[r+direction]-pred_index_list[r])
         else:
            dist = self.array_length-abs(pred_index_list[(r+direction)%n]-pred_index_list[r])
         move = min(abs(dist)-1, self.max_move)*direction
         self.pred_list[pred_index_list[r]]=0
         self.pred_list[(pred_index_list[(r)]+move)%self.array_length] = 1


   def spawn_new_prey(self, i):
      """Checks if there are empty slots adjacent to i. If yes, spaws a new prey there."""

      r = random.choice([-1,1])
      if self.prey_list[(i+r)%self.array_length]==0:
         self.prey_list[(i+r)%self.array_length] = 1
      elif self.prey_list[(i-r)%self.array_length]==0:
         self.prey_list[(i-r)%self.array_length] = 1
         
   def spawn_new_pred(self, i):
      """Checks if there are empty slots adjacent to i. If yes, spaws a new predator there."""
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
      self.interaction() #pred health->1, prey->0 when interacting
      self.update_rep_death() #increase prey health by r_x, reduce pred health by r_y
      self.move_all_pred() #every pred moves

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
#    plt.xlim(300,500)
#    plt.ylim(600,800)
    plt.title("Preys Vs Predators for all Time-Steps")
    #c=timesteps colorcodes the scatterplot so that earlier datapoint are violet and newer are yellow
    plt.scatter(num_pred_list, num_prey_list, c=timesteps)

def plot_pred(num_prey_list, num_pred_list, MA=1):

   if MA != 1:
      for i in range(len(num_prey_list)-MA):
         num_prey_list[i] = sum(num_prey_list[i:i+MA])/(MA)
         num_pred_list[i] = sum(num_pred_list[i:i+MA])/(MA)
      del num_prey_list[i:]

   fig, ax = plt.subplots()
   ax.plot(num_pred_list[200:], 'r-')
   ax.set_xlabel('time')
   ax.set_ylabel('Prey population', color='red')

   ax2=ax.twinx()
   ax2.plot(num_prey_list[200:], 'b-')
   ax2.set_ylabel('Predator population', color='blue')
   plt.show()

def linear_function(x,a,b):
   return a+b*x

def diff_plot(num_prey_list, num_pred_list,start, mode=1):
   diff_list=[]
   for i in range(len(num_pred_list)-1):
      diff_list.append(num_pred_list[i+1]-num_pred_list[i])

   x = np.linspace(min(num_prey_list[start:]),max(num_prey_list[start:]), 100) #time[:num*2]

   init_vals = [1,1]
   best_vals, covar = curve_fit(linear_function, num_prey_list[start:-1], diff_list[start:], p0=init_vals)
   y = [linear_function(i, best_vals[0], best_vals[1]) for i in x]
   if mode==1:
      fig, ax = plt.subplots()
      ax.plot(x,y,'b-')
      ax.plot(num_prey_list[start:-1],diff_list[start:], 'ro')
      ax.set_xlabel('Prey population')
      ax.set_ylabel('Predtor population growth', color='red')
   
   #   ax2=ax.twinx()
   #   ax2.plot(diff_list[200:], 'b-')
   #   ax2.set_ylabel('Change in Prey population', color='blue')
      plt.show()
   
   return best_vals

def save_data(k, prey_num, pred_num, msd=0   ):
   dirname = os.path.dirname(__file__)
   filename = os.path.join(dirname, 'data_save')
   f = open(filename, "a+")
   f.write("{} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(array_length, r_x, r_y, max_move, 
                                              k, prey_num, pred_num, msd))
   f.close()

#def log_pop(num_prey_list, num_pred_list):
def noise_approximation(N,L,n):
   sum_of_rmse = []
   for i in range(n):
      random_list = [0 for iter in range(L-N)]+[1 for iter in range(N)]
      random.shuffle(random_list)
      sum_of_rmse.append(meanSqIterator(random_list))
      
   return np.average(sum_of_rmse)

def meanSqIterator(a_list):
       """Returns the mean square distance of the preys/predators from the provided list."""
       sumOfMeanSq = 0
       n_pred = sum(a_list)
       N = len(a_list)
       index_list = [index for index, value in enumerate(a_list) if value==1]
       
       for i in range(n_pred-1):
           sumOfMeanSq += (index_list[i+1] - index_list[i])**2
       sumOfMeanSq += (abs(index_list[-1]-index_list[0]-N))**2
       
       RMSE = (sumOfMeanSq/n_pred)**0.5
       return RMSE

def nrmse(pred_list):
   rmse = meanSqIterator(pred_list)
   n = 0
   n_pred = sum(pred_list)
   N = len(pred_list)
   noise_approx = noise_approximation(n_pred, N, 30)
   
   nrmse = (rmse-noise_approx)/(((1/n_pred*(n_pred-1+(N-n_pred+1)**2))**0.5)-noise_approx)
   
   return nrmse

def main():
   seedlist = [iter for iter in range(1)]
   save_list = []
#   noice_approx = noise_approximation(pred_num,array_length,100)
   
   for seed in seedlist:
      a = Environment()        #create instance of the class
      a.init(seed)
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
#      mse_all = [nrmse(x) for x in predator_list]
#      mse_avg = np.average(mse_all[start:])
#      prey_avg = np.average(num_prey[start:])
#      pred_avg = np.average(num_pred[start:])
#      k = diff_plot(num_prey, num_pred,start, 0)[1]
#      save_data(k,prey_avg, pred_avg, mse_avg)
      #diplay the data
      plotting_population(num_pred, num_prey)
      plotting_system(prey_list, 'Preys')
      plotting_system(predator_list, 'Predators')


if __name__ == "__main__":
   main()
