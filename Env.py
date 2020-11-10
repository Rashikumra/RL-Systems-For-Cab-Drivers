# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        '''tuple of pickup and drop location , also driver has a option to go offline(0,0) '''
        self.action_space = [[x,y] for x in range(5) for y in range(5) if (x!=y) or ((x==0) and (y==0))] 
        
        '''state space -defined by X driver's current location along with the T time components (hour of the day) & D day of the          sweek'''
        self.state_space = [[X,T,D] for X in range(5) for T in range(24) for D in range(7)]
        
        '''A random initilization of any state'''
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector                format. Hint: The vector is of size m + t + d."""
        ## create a list of 0 of length m+t+d
        state_encod = np.zeros(m+t+d)
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
  
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   

    '''new function change name '''
    def update_to_newtime(self,time,day,time_taken):
        updated_day_time= time + math.ceil(time_taken)
        updated_weekday = day
        if updated_day_time > 23:
            updated_day_time = updated_day_time % 24
            updated_weekday += 1
            if updated_weekday > 6:
                updated_weekday = updated_weekday % 7
        #print("New_time ",new_time_of_day,new_day_of_week)
        return updated_day_time,updated_weekday
    
    
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        reward = 0
        driver_curr_loc = state[0]
        pickup_location = action[0]
        drop_location = action[1]
        time_of_day = state[1]
        weekday = state[2]
        updated_day_time = time_of_day 
        updated_weekday = weekday 
        
        if driver_curr_loc!=pickup_location: #Calculate the new time of day and week when cab reaches pickup position
            time_taken = Time_matrix[driver_curr_loc][pickup_location][time_of_day][weekday]
            updated_day_time,updated_weekday = self.update_to_newtime(time_of_day,weekday,time_taken) 
        
        if (pickup_location == 0) and (drop_location==0):
            reward = -C
        else:
            #print(pickup_pos,drop_pos,new_time_of_day,new_day_of_week)
            reward = R*Time_matrix[pickup_location][drop_location][updated_day_time][updated_weekday] - C*(Time_matrix[pickup_location][drop_location][updated_day_time][updated_weekday] + Time_matrix[driver_curr_loc][pickup_location][time_of_day][weekday])
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        driver_curr_loc= state[0]
        pickup_location = action[0]
        drop_location = action[1]
        time_of_day = state[1]
        weekday = state[2]
        updated_day_time = time_of_day #variable to calculate the time when cab reached pickup posiion if curr_pos != pickup_pos
        updated_weekday = weekday #variable to calculate the day of week when cab reached pickup posiion if curr_pos != pickup_pos
        total_time = 0
        if driver_curr_loc!=pickup_location: #Calculate the new time of day and week when cab reaches pickup position
            time_taken = Time_matrix[driver_curr_loc][pickup_location][time_of_day][weekday]
            updated_day_time,updated_weekday = self.update_to_newtime(time_of_day,weekday,time_taken)
            total_time += time_taken
        
        if (pickup_location == 0) and (drop_location==0):
            total_time += 1
            updated_day_time,updated_weekday = self.update_to_newtime(time_of_day,weekday,1)
            next_state = [driver_curr_loc,updated_day_time,updated_weekday]
        else:
            time_taken = Time_matrix[pickup_location][drop_location][updated_day_time][updated_weekday]
            total_time += time_taken
            final_time_of_day,final_day_of_week = self.update_to_newtime(updated_day_time,updated_weekday,time_taken)
            next_state = [drop_location,final_time_of_day,final_day_of_week]
        return next_state,total_time




    def reset(self):
        return self.action_space, self.state_space, self.state_init
