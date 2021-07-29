# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:06:18 2021

@author: HP
"""


import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import operator
import gc
from scipy.spatial.distance import *

from geopy import distance

# check average latency 
# latency = time from sensing to transferring to the BS
def check_latency() :
    global entities, high_TTL, med_TTL, low_TTL
    latency = 0
    for i in range(len(entities[eG+mG+1].recBuf)) :
        
        item = entities[eG+mG+1].recBuf[i]
        max_TTL = 0
        
        # 1 = high priority
        if item[3]==1 :
            max_TTL = high_TTL
            #print("high")
            #print("max ttl "+str(max_TTL))
        elif item[3] == 2 :
            max_TTL = med_TTL
            #print("med")
            #print("max ttl "+str(max_TTL))
        elif item[3] == 3:
            max_TTL = low_TTL
            #print("low")
            #print("max ttl "+str(max_TTL))
        #print(item[4])
        latency += (max_TTL - item[4])
        #print(str(latency))
    avg_latency = (latency/len(entities[eG+mG+1].recBuf))
    print("total latency : "+str(latency))
    print("average latency : "+str(avg_latency))
        

# check energy status of the nodes
def check_energy_status() :
    global eG, mG, entities, energy_fog, energy_mobile
    energy_drain = 0
    
    i=0
    while i < eG :
        energy_drain = energy_drain + (energy_fog - entities[i].rE)
        i += 1
    
    while i < eG+mG :
        energy_drain = energy_drain + (energy_mobile - entities[i].rE)
        i += 1
    
    return energy_drain
    #print("total energy drain : "+str(energy_drain))
        

# check if same event id present or not
def check_id_not_present(L,id) :
    mark = True
    for item in L :
        if item[0] == id :
            mark = False
            break
    return mark

#produce location of the node within simulation area
def location() :
    global Xlim, Ylim
    loc = [random.uniform(Xlim[0],Xlim[1]),random.uniform(Ylim[0],Ylim[1])]
    return loc

#damage fog nodes only due to aftershocks
#depends on distance from epicentre
def damage() :
    global eG,damaged_station,alive_station,max_node_distance
    
    #r = random.random()
    # Number controlling damage
    R = 0.20
    
    # Distances of alive node from epicenter
    distances = {i: find_dist(entities[i].my_coor[0],entities[i].my_coor[1] , epicentre[0], epicentre[1]) for i in range(eG)}
    
    # Maximum distance
    maxd = max(list(distances.values()))
    
    
    #set energy of the node to zero to make them disabled
    for i in range(eG) :
        d = float(distances[i]) / float(maxd)
       
        if(d < R) :
            entities[i].rE = 0
        if(entities[i].rE == 0) :
            damaged_station += 1
            alive_station -= 1

#calculate distance between two points              
def find_dist(x1, y1, x2, y2):

    global mTOm, lat_dist, lon_dist
    return math.sqrt(math.pow(x2 - x1, 2) * lat_dist + math.pow(y2 - y1, 2) * lon_dist) * mTOm
   
#calculate fog node's distance from epicentre
def distance_from_epicentre() :
    global epicentre, entities, node_distance
    #print("epicentre : ",epicentre)
    
    #distance of each node from epicentre
    for i in range(eG):
        node_distance[i] = find_dist(entities[i].my_coor[0], entities[i].my_coor[1], epicentre[0], epicentre[1])
  
#mobility of the mobile node's location
def latp(x,y):
    
    #displacement range
    r = 0.01
    
    global Xlim, Ylim
    while True :
        x_loc = random.uniform(x-r,x+r)
        y_loc = random.uniform(y-r,y+r)
        
        if((x_loc >= Xlim[0] and x_loc<= Xlim[1]) and (y_loc >= Ylim[0] and y_loc <= Ylim[1])) :
            break

    return (x_loc,y_loc)
    
#before sending data to next hop, first sort the events by their priority
#high priority to low priority
#low TTL to high TTL
def sort_events_by_priority(L) :
    L = sorted(sorted(L, key = lambda x: (x[4])), key = lambda y : (y[3]))
    return L

#fog nodes' status, alive or dead
def show_plot_1() : 
    global entities, epicentre, Xlim, Ylim
    
    alive_fog_location_x = []
    alive_fog_location_y = []

    dead_fog_location_x = []
    dead_fog_location_y = []

    #current location of the nodes
    for i in range(eG):
        if entities[i].rE >= baseE   :
            alive_fog_location_x.append(entities[i].my_coor[0])
            alive_fog_location_y.append(entities[i].my_coor[1])
        else :
            dead_fog_location_x.append(entities[i].my_coor[0])
            dead_fog_location_y.append(entities[i].my_coor[1])

    alive_mob_loc_x = []
    alive_mob_loc_y = []
    dead_mob_loc_x = []
    dead_mob_loc_y = []
    i = eG
    while i< eG + mG :
        if entities[i].rE >= baseE :
            alive_mob_loc_x.append(entities[i].my_coor[0])
            alive_mob_loc_y.append(entities[i].my_coor[1])
        else :
            dead_mob_loc_x.append(entities[i].my_coor[0])
            dead_mob_loc_y.append(entities[i].my_coor[1])
        i +=1

    #plot the current location
    plt.xlim(Xlim[0], Xlim[1])
    plt.ylim(Ylim[0], Ylim[1])

    plt.scatter(alive_fog_location_x, alive_fog_location_y, color = 'green', label = "alive fogs")
    plt.scatter(dead_fog_location_x, dead_fog_location_y, color = 'red', label = 'dead fogs')
    plt.scatter(BC[0], BC[1], color = "blue", label = "base station")
    #plt.scatter(alive_mob_loc_x, alive_mob_loc_y, color = 'blue', label = 'alive mobiles')
    #plt.scatter(dead_mob_loc_x, dead_mob_loc_y, color = 'black', label = 'dead mobiles')
    plt.scatter(epicentre[0], epicentre[1], color='yellow', label = "epicentre")
    plt.show()    

# Event generation status
# no of high priority, med priority and low priority events
def show_plot_2() :
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    events_priority = ['high', 'medium', 'low']
    no_of_events = [no_high_events, no_med_events, no_low_events]
    ax.bar(events_priority,no_of_events)
    plt.show()
    
# data reached to Base station
# ratio data delivery rate for each type events
def show_plot_3() :
    global entities, eG, mG
    global no_high_events, no_med_events, no_low_events
    print("plot 3 function is called")
    high = 0
    med = 0
    low = 0
    
    print("Base station's total events in buffer :"+str(len(entities[eG+mG+1].recBuf)))
    if len(entities[eG+mG+1].recBuf) > 0 :
        for i in entities[eG+mG+1].recBuf :
            #print(i)
            if(i[3]==1) :
                high += 1
            elif (i[3]==2) :
                med +=1
            elif (i[3] == 3) :
                low += 1
                
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    events_priority = ['high', 'medium', 'low']
    events_reached = [(high / no_high_events), (med / no_med_events),(low /  no_low_events)]
    ax.bar(events_priority,events_reached)
    plt.show()
    
def calculate_PDR(priority_id) :
    global entities, eG, mG
    global no_high_events, no_med_events, no_low_events
    x = 0
    #high = 0
    #med = 0
    #low = 0
    id = priority_id
    
    if len(entities[eG+mG+1].recBuf) > 0 :
        for i in entities[eG+mG+1].recBuf :
            #print(i)
            if(i[3]==id) :
                x += 1
            """
                high += 1
            elif (i[3]==2) :
                med +=1
            elif (i[3] == 3) :
                low += 1
    pdr_now = ((high / no_high_events), (med / no_med_events),(low /  no_low_events))
    """
    if id == 1 :
        pdr_now = x / no_high_events
    elif id == 2 :
        pdr_now = x / no_med_events
    elif id == 3 :
        pdr_now = x / no_low_events
        
    return pdr_now

def check_duplicate_in_bs() :
    a_list = []

    for item in entities[eG+mG+1].recBuf :
        a_list.append(item[0])
    contains_duplicates = any(a_list.count(element) > 1 for element in a_list)
    if(contains_duplicates) :
        print("duplicate event present in buffer")
    else :
        print("no duplicate event in buffer")
   
def check_if_any_event_in_bs_ttl_less_zero() :
    flag_lz = False
    
    for item in entities[eG+mG+1].recBuf :
        if item[4] < 0 :
            flag_lz = True
            break
    
    if(flag_lz) :
        print("error")
        
    else :
        print("no error")
        
class Node(object):
    def __init__(self, env, ID, my_coor):
        
        global T

        self.ID = ID
        self.env = env

        # Neighbor list
        self.nlist = []

        self.old_coor = None
        
        # Node's current location
        self.my_coor = my_coor

        self.start = True
        
        # Buffer to store data
        self.recBuf = []

        # Time instant
        self.ti = random.randint(0, PT - 1)

        # List of events detected by system
        self.buffer = []

        # Fog's view of the network
        self.myG = nx.Graph()

        # Dictionary of fog node coordinates
        self.FC = {u: None for u in range(eG)}

        # Node motif centrality
        # Node remaining store capacity
        # Node's activeness
        # Node's distance from base station
        self.NMC = {int(self.ID[2:]): (0,0,1,0)}
        
        # shortest path to base station
        self.path_present = False
        self.shortest_path_route = []
        
        #activeness of the node
        self.activeness = 0
        
        # Find neighbors initially
        self.scan_neighbors()

        # List of new nodes in the neighborhood
        self.JReqs = []

        # List of events detected by system
        self.events = []

        if 'E-' in self.ID:
            #self.my_waypoint = waypoints[int(self.ID[2:])]
            self.rE = energy_fog
            #self.rE = 150.0
            #self.recBuf = simpy.Store(env, capacity = recBufferCapacity_fog)
            #self.recBuf = simpy.Store(env)
            self.temp = None
            self.next_hop = None
            
            #self.env.process(self.sense())
            self.env.process(self.send())
            self.env.process(self.receive())
            #self.env.process(self.printBuf())

        if self.ID == 'E-0':
            self.globalG = nx.DiGraph()
            #self.env.process(self.find_global_graph())
            self.env.process(self.time_increment())
            self.env.process(self.event_TTL_update())

        if 'EG' in self.ID:
            self.env.process(self.genEvent())

        if 'M-' in self.ID:
            
            self.rE = energy_mobile
            #self.recBuf = simpy.Store(env)
            #self.recBuf = simpy.Store(env, capacity = recBufferCapacity_mobile)
            
            self.next_hop = None
            
            self.active = int(self.ID[2:]) % 5 + 1
            self.prompt = 1

            self.preference = 0
            self.observations = 1

            self.env.process(self.sense())
            self.env.process(self.send())
            self.env.process(self.receive())
            #self.env.process(self.printBuf())
            
        if 'BS-' in self.ID :
            self.rE = 10000
            #self.recBuf = simpy.Store(env)
            #self.env.process(self.printBuf())
    
    def find_shortest_path(self) :
        global entities, eG, mG
        print("function find_shortest_path() called :"+self.ID[2:])
        #print(nx.is_directed(entities[0].globalG))
        
        print(entities[0].globalG.nodes)
        
        #self.shortest_path_route = nx.shortest_path(entities[0].globalG, self.ID[2:], eG+mG+1 )
        #print(self.shortest_path_route)
        
        
        self.path_present = nx.has_path(entities[0].globalG, source = int(self.ID[2: ]), target = eG+mG+1)
        print(self.ID+" path present or not :"+str(self.path_present))
        if (self.path_present) :
             self.shortest_path_route = nx.shortest_path(entities[0].globalG, int(self.ID[2: ]), eG+mG+1 )
        else :
            self.shortest_path_route = []
        
            
    def event_TTL_update(self):
        global minimumWaitingTime, recBufferCapacity
        while True:
            if T>0 and T % PT == 0:
                if self.ID =="E-0" :
                    # for all nodes (fog + mobile) eG+mG
                    # Update TTL and remove event packets with TTL <= 0
                    
                    for i in range(eG+mG):
                        for j in range(len(entities[i].recBuf)) :
                            item =  entities[i].recBuf[j]
                            item = [item[0], item[1], item[2], item[3], item[4]-1]
                            entities[i].recBuf[j] = item
                            
                        entities[i].recBuf = [event for event in entities[i].recBuf if event[4] > 0]
                        entities[i].recBuf = entities[i].recBuf[-recBufferCapacity:]
                        
            yield self.env.timeout(minimumWaitingTime)  
        
        
    def genEvent(self):

        global T, Duration, frequencyEvent, globalEventCounter, Xlim, Ylim
        global no_high_events, no_med_events, no_low_events 
        
        while True :
            
            # Generate new events
            #if T<=40 and T % frequencyEvent == 0:
            if T % frequencyEvent == 0:
                for i in range(how_many_events):

                    # Random event
                    #fewer events with higher priority
                    
                    x = random.uniform(0.0,1.0)
                    if((x >= 0.0) and (x <= 0.15)) :
                        priority = 1
                        no_high_events +=1
                        TTL = 12
                        
                    elif ((x > 0.15) and (x <= 0.5)) :
                        priority = 2
                        no_med_events +=1
                        TTL = 6
                        
                    elif((x > 0.5) and (x <= 1.0)) :
                        priority = 3
                        no_low_events +=1
                        TTL = 3
                        
                    # Time to live by hop counts
                    #TTL = 10
                    
                    new_event = [globalEventCounter, (random.uniform(Xlim[0], Xlim[1]), random.uniform(Ylim[0], Ylim[1])), T, priority, TTL]
                    
                    # Saved event
                    # CREATE 10 COPIES OF EACH EVENT
                    for i in range(5) :
                        self.events.append(new_event)
                    
                    globalEventCounter += 1

            # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
            self.events = self.updateEventList1(self.events)

            yield self.env.timeout(minimumWaitingTime)
            
    def updateEventList1(self, L):

        global frequencyEvent, T, recur

        remove_indices = []
        for each in L:
            if each[2] <= T - recur:
                remove_indices.append(L.index(each))

        return [i for j, i in enumerate(L) if j not in remove_indices]
    
    def updateEventList2(self,L):
        
        remove_indices = []
        L_dash = []
        # count = 0
        for i in range(len(L)):
            if L[i][4] <= 0:
                #count += 1
                remove_indices.append(i)
                
        """
        if count > 0 :
            print("at t = "+str(T)+" removed no. of elements = "+str(count)+" from node = "+str(self.ID))
        """
        L_dash = [i for j, i in enumerate(L) if j not in remove_indices]
        return L_dash
    
    def sense(self):

        global baseE, L, sensing_range_mobile, eG, mG, senseE, prompt_incentive, Tot
        while self.rE > baseE:

            if T % frequencyEvent == 0:
                self.rE = self.rE - senseE
                
                # Sense event in the vicinity
                for each in entities[eG + mG].events:
                    
                    if find_dist(each[1][0], each[1][1], self.my_coor[0], self.my_coor[1]) <= sensing_range_mobile:
                        
                        #if random.choice([0, 1]) == 1:
                        if True :
                            self.recBuf.append(each)
                            Tot.append(each)
                            self.prompt += prompt_incentive
                        else:
                            self.prompt -= prompt_incentive
                            self.prompt = max(1, self.prompt)

                # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
                #self.events = self.updateEventList1(self.events)
                self.move()

            yield self.env.timeout(minimumWaitingTime)
      
    def scan_neighbors(self):
        
        global eG, sensing_range, sensing_range_mobile, entities, D, scanE, T
        
        self.nlist = []
        
        if 'E-' in self.ID:
            
            if T > 1:
                self.rE = self.rE - scanE
                
            self.myG = nx.Graph()
            self.myG.add_node(int(self.ID[2:]))
            
            if self.start:
                for u in range(eG + mG):
                    next_coor = location()
                    if find_dist(self.my_coor[0], self.my_coor[1], next_coor[0], next_coor[1]) <= sensing_range:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

                self.start = False
            
            else:
                for u in range(eG + mG):
                    if find_dist(self.my_coor[0], self.my_coor[1], entities[u].my_coor[0], entities[u].my_coor[1]) <= sensing_range:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

            self.nlist = [u for u in self.nlist if u != int(self.ID[2:])]
           
        elif 'M-' in self.ID :
            if T > 1:
                self.rE = self.rE - scanE
                
            self.myG = nx.Graph()
            self.myG.add_node(int(self.ID[2:]))
            
            if self.start:
                for u in range(eG + mG):
                    next_coor = location()
                    if find_dist(self.my_coor[0], self.my_coor[1], next_coor[0], next_coor[1]) <= sensing_range_mobile:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

                self.start = False
            
            else:
                for u in range(eG + mG):
                    if find_dist(self.my_coor[0], self.my_coor[1], entities[u].my_coor[0], entities[u].my_coor[1]) <= sensing_range_mobile:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

            self.nlist = [u for u in self.nlist if u != int(self.ID[2:])]
    
    def move(self):

        global Xlim, Ylim, baseE

        if self.rE > baseE:

            if 'M-' in self.ID:
                self.old_coor = self.my_coor
                
                x_loc = self.old_coor[0]
                y_loc = self.old_coor[1]
                # LATP
                self.my_coor = latp(x_loc, y_loc)

        self.scan_neighbors()
        
    def find_NMC(self):
        
        self.NMC = {u: (0,0,0,0) for u in self.myG.nodes()}
        self.NMC[int(self.ID[2:])] = (0,0,0,0)
        
        # calculate motif centrality
        NMC_temp = {u: 0 for u in self.myG.nodes()}
        NMC_temp[int(self.ID[2:])] = 0
        
        for u in self.myG.nodes():
            for v in self.myG.nodes():
                if v <= u:
                    continue

                for w in self.myG.nodes():
                    if w <= v:
                        continue
                    if self.myG.has_edge(u, v) and self.myG.has_edge(v, w) and self.myG.has_edge(u, w):
                        NMC_temp[u] += 1
                        NMC_temp[v] += 1
                        NMC_temp[w] += 1
                        
        for i in self.myG.nodes() : 
            if i < eG :
                remaining_capacity = recBufferCapacity_fog - len(entities[i].recBuf)
            elif i< (eG+mG) :
                remaining_capacity = recBufferCapacity_mobile - len(entities[i].recBuf)
            
            actv_score = entities[i].activeness
            nmc_distance = find_dist(entities[i].my_coor[0],entities[i].my_coor[1], BC[0], BC[1])
            
            self.NMC[i] = (NMC_temp[i], remaining_capacity, actv_score,nmc_distance)
        
    def send(self) :
        global T, pr_fg_E, fg_fg_E, baseE, V1, V2, Rec
        
        while True:
            if self.rE > baseE:
                
                
                """
                # Send a leave message (LM) to all current neighbors asking
                # them to remove me from their neighbor lists
                if T % PT == 0 and not self.start:
                    for u in self.nlist:
                        entities[u].buffer.append(['LM', int(self.ID[2:])])
                        self.myG = nx.Graph()
                        self.myG.add_node(int(self.ID[2:]))
                        self.nlist = []

                    #self.move() 
        
                # Send a join request (JReq) to the new neighbor nodes
                # informing them of my existence
                if T % PT == (self.ti + 1) % PT:
                    self.scan_neighbors()
                    for u in self.nlist:
                        entities[u].buffer.append(['JReq', (int(self.ID[2:]), self.my_coor)])

                # Send a join response (JRes) to the new neighbor node
                # informing them about my neighbors
                if T % PT == (self.ti + 2) % PT:
                    for m in self.JReqs:
                        entities[m[1][0]].buffer.append(['JRes', int(self.ID[2:]), [(u, entities[u].my_coor) for u in self.nlist]])

                # Send a update request message (UM) to all current neighbors
                # informing them of my neighbors
                if T % PT == (self.ti + 3) % PT:
                    for u in self.nlist:
                        entities[u].buffer.append(['UM', int(self.ID[2:]), [(u, entities[u].my_coor) for u in self.nlist]])
                """
                if T % PT == 0 and not self.start:
                    self.move()
                
                if T % PT == (self.ti + 4) % PT:
                    self.find_shortest_path()
                    #if (random.uniform(0,1) <= 0.5) :
                    if True :
                        # send data to next hop or base station
                        # sort buffer by priority
                        # sort the events acccording to their priority
                        self.recBuf = sort_events_by_priority(self.recBuf)
                        
                        # Send data to next gop or base station
                        if len(self.recBuf) > 0 :
                            if self.path_present :
                                self.next_hop = self.shortest_path_route[1]
                                """
                                next_hop_capacity = 0
                                
                                if self.next_hop < eG :
                                    next_hop_capacity = recBufferCapacity_fog - len(entities[self.next_hop].recBuf)
                                elif self.next_hop < eG + mG :
                                    next_hop_capacity = recBufferCapacity_mobile - len(entities[self.next_hop].recBuf)
                                
                                for i in range(next_hop_capacity) :
                                    if len(self.recBuf) > 0 :
                                        item = self.recBuf.pop(0)
                                        entities[self.next_hop].recBuf.append(item)
                                        self.rE -= fg_fg_E
                                """
                                
                                if self.next_hop == eG+mG+1 :
                                    
                                    while len(self.recBuf) > 0 :
                                        item = self.recBuf.pop(0)
                                        if check_id_not_present(entities[self.next_hop].recBuf,item[0]):
                                            entities[self.next_hop].recBuf.append(item)
                                            self.rE -= fg_fg_E
                                        else :
                                            continue
                                    
                                else :
                                    next_hop_capacity = 0
                                
                                    if self.next_hop < eG :
                                        next_hop_capacity = recBufferCapacity_fog - len(entities[self.next_hop].recBuf)
                                    elif self.next_hop < eG + mG :
                                        next_hop_capacity = recBufferCapacity_mobile - len(entities[self.next_hop].recBuf)
                                
                                    for i in range(next_hop_capacity) :
                                        if len(self.recBuf) > 0 :
                                            item = self.recBuf.pop(0)
                                            entities[self.next_hop].recBuf.append(item)
                                            self.rE -= fg_fg_E
                            else :
                                break
                            
                        
            yield self.env.timeout(minimumWaitingTime)
     
    def receive(self):

        global T, eT, V1
        
        while True:
            
            # When you receive a leave message
            # update your neighbor list
            LMs = [m for m in self.buffer if 'LM' in m]
            self.myG.remove_nodes_from([m[1] for m in LMs if m[1] in self.myG.nodes()])

            self.buffer = [m for m in self.buffer if m not in LMs]
            
            # Accept a join request (JReq) from a new neighbor
            # and update G
            self.JReqs = [m for m in self.buffer if 'JReq' in m]
            
            for m in self.JReqs:
                self.myG.add_edge(int(self.ID[2:]), m[1][0])
                self.FC[m[1][0]] = m[1][1]

            self.buffer = [m for m in self.buffer if m not in self.JReqs]
            
            # When I receive a join response (JRes) message from a neighbor
            # containing his neighbor information, I update my graph
            JRess = [m for m in self.buffer if 'JRes' in m]
            for m in JRess:
                self.myG.add_edges_from([(m[1], u) for (u, c) in m[2]])

                for (u, c) in m[2]:
                    self.FC[u] = c

            self.buffer = [m for m in self.buffer if m not in JRess]
            
            # When I receive a update request message (UM) from a neighbor
            # I delete all his prior neighbors of that node and add his current neighbors to the graph
            UMs = [m for m in self.buffer if 'UM' in m]
            for m in UMs:

                # Update graph with new 2 hop neighbor information
                if m[1] in self.myG.nodes():
                    self.myG.remove_node(m[1])
                self.myG.add_node(m[1])

                for (u, c) in m[2]:
                    self.myG.add_edge(m[1], u)
                    self.FC[u] = c
            
            self.buffer = [m for m in self.buffer if m not in UMs]
            
            if self.rE > baseE :
                self.find_NMC()
            
            yield self.env.timeout(minimumWaitingTime)       
   
    
    def find_global_graph(self):

        global sensing_range, sensing_range_mobile, eG, mG
        #print("global graph start")
        self.globalG = nx.DiGraph()
        
        
        L = [u for u in range(eG+mG) if entities[u].rE > baseE]
        #print(L)
        self.globalG.add_nodes_from(L)
        self.globalG.add_node(eG+mG+1)

        #print("graphs nodes :")
        #print(self.globalG.nodes)
        #print(self.globalG.edges)
        #print(L)
        for u in L:
            
            for v in L:
                if u == v:
                    continue
                
                elif u < eG :
                    
                    if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[v].my_coor[0], entities[v].my_coor[1]) <= sensing_range:
                        self.globalG.add_edge(u, v)
                
                elif u < eG+mG :
                    if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[v].my_coor[0], entities[v].my_coor[1]) <= sensing_range_mobile:
                        self.globalG.add_edge(u, v)

            if u < eG :
                if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[eG+mG+1].my_coor[0], entities[eG+mG+1].my_coor[1]) <= sensing_range:
                    self.globalG.add_edge(u, eG+mG+1)
            elif u < eG+mG :
                if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1]) <= sensing_range_mobile:
                    self.globalG.add_edge(u, eG+mG+1)
                
        #print(self.globalG.edges)
    """
    
    def find_global_graph(self):

        global sensing_range, eG, baseE
        self.globalG = nx.Graph()
        self.globalG.add_node(2000)
        L = [u for u in range(eG) if entities[u].rE > baseE]
        self.globalG.add_nodes_from(L)

        for u in L:
            for v in L:
                if v <= u:
                    continue
                if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[v].my_coor[0], entities[v].my_coor[1]) <= sensing_range:
                    self.globalG.add_edge(u, v)

            if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1]) <= sensing_range:
                self.globalG.add_edge(u, 2000)
     """
    
    def time_increment(self):

        global T, Tracker, eG, Correlate, G, tm, V1, V2, eT
        #pdr_calculated = ()
        while True:

            T = T + 1
            tm = tm + 0.002
            self.find_global_graph()

            if T > 2:
                Tracker.append([(entities[u].my_coor[0], entities[u].my_coor[1], entities[u].NMC[u]) for u in range(eG)])

            # Latency
            Z = list(entities[eG + mG + 1].recBuf)
            t = len(Z)
            Z = Z[eT + 1:]
            V1.extend([T - m[-1] for m in Z])
            eT = t - 1

            print("current time : "+str(T))
            
            
            
            if T %10 == 0 :
                
                """
                print("elements in base station at time = "+str(T))
                print(entities[eG+mG+1].recBuf.items)
                """
                
                pdr_h = calculate_PDR(1)
                pdr_high.append(pdr_h)
                
                pdr_m = calculate_PDR(2)
                pdr_med.append(pdr_m)
                
                pdr_l = calculate_PDR(3)
                pdr_low.append(pdr_l)
                
                energy_file.append(check_energy_status())
            """
            for i in range(eG + mG) :
                if len(entities[i].recBuf) > 0 :
                    print("NODE - "+str(i)+" Buffer :")
                    print(entities[i].recBuf)
            
            
            print ("Base Station buffer : ")
            print(entities[eG+mG+1].recBuf.items)
               """ 
            yield self.env.timeout(minimumWaitingTime) 
            
#co-ordinates of fog nodes to be selected from here
Coordinates = {0: (40.0035772010397, 73.06123207638116), 1: (40.00618211979722, 73.03972661166507), 2: (40.013210802626034, 73.09796817350032), 3: (40.00210141356536, 73.15918987314448), 4: (40.01250883188785, 73.07970690993588), 5: (40.01071833931967, 73.1571427803427), 6: (40.012096207524415, 73.03591575987609), 7: (40.002124765303364, 73.10092583696022), 8: (40.00366424001856, 73.1333772887022), 9: (40.01454705271114, 73.06367244622726), 10: (40.016932313172596, 73.04904734206828), 11: (40.01442782871262, 73.00408125675949), 12: (40.01477122188726, 73.15609325691526), 13: (40.01420327608633, 73.1184319836396), 14: (40.005431055718496, 73.14662901935398), 15: (40.01454195183018, 73.10894678030704), 16: (40.0141866218414, 73.14954483235424), 17: (40.00979442577592, 73.09641565411026), 18: (40.00417943198239, 73.00859216856146), 19: (40.016451363117994, 73.10427638191584), 20: (40.00939638652993, 73.15614652591074), 21: (40.01472797067223, 73.16701901552486), 22: (40.01205587311622, 73.0172807155463), 23: (40.01623934480377, 73.12749328271191), 24: (40.00650776994118, 73.07067330957713), 25: (40.01489326238527, 73.16182423311713), 26: (40.00665281347467, 73.10474522399561), 27: (40.004585200895185, 73.1497061921605), 28: (40.01433839958769, 73.0066649008898), 29: (40.00597746425146, 73.02382155904463), 30: (40.00936925378166, 73.14721958162579), 31: (40.011947488633936, 73.07988925768159), 32: (40.00568038394408, 73.1303056900752), 33: (40.01389068923644, 73.01975045710142), 34: (40.01380874633281, 73.14448823245999), 35: (40.01330813888446, 73.16129604315337), 36: (40.00152601949556, 73.07571155761097), 37: (40.01668063351763, 73.00021331495948), 38: (40.01440259971463, 73.13308957495413), 39: (40.0039697761366, 73.14931776042789), 40: (40.01459935997943, 73.13161290991592), 41: (40.00965493559254, 73.06733126742914), 42: (40.01023729061635, 73.11850489247747), 43: (40.00087303032123, 73.14345541488663), 44: (40.00734776850684, 73.10530212717642), 45: (40.015506179231444, 73.13234179311026), 46: (40.00729986319626, 73.1195664591611), 47: (40.00317001341289, 73.11947274663572), 48: (40.01474779823594, 73.147022802688), 49: (40.001374674056436, 73.0371232462076), 50: (40.00007878209582, 73.08611450584546), 51: (40.00867494743175, 73.06742567599802), 52: (40.00526202405532, 73.12124174407218), 53: (40.003879154749505, 73.00985015829302), 54: (40.01539750499658, 73.0152767990886), 55: (40.012656550422335, 73.00367595730799), 56: (40.0016095727478, 73.10368973737435), 57: (40.01334876560259, 73.16675702980912), 58: (40.008634552938595, 73.00620328362038), 59: (40.015302620280515, 73.15936570023757), 60: (40.006522085663576, 73.15610364122618), 61: (40.00590819328866, 73.03918231671747), 62: (40.001084330451185, 73.10055510953825), 63: (40.004803699876575, 73.13539526552472), 64: (40.00194417798886, 73.01898987345501), 65: (40.00970865892598, 73.04895256766513), 66: (40.00924588377529, 73.06896563044849), 67: (40.01535216008005, 73.10079095220422), 68: (40.01508603243717, 73.17251835742644), 69: (40.00033301789189, 73.04977017759654), 70: (40.00428847964569, 73.02811201535384), 71: (40.00615322805891, 73.14982741495822), 72: (40.009565895350825, 73.03243382900442), 73: (40.00883517114237, 73.08578462871995), 74: (40.0147586737026, 73.15301617916742), 75: (40.013937276876725, 73.16891716145342), 76: (40.00946419013778, 73.12365020352568), 77: (40.01314430440763, 73.08898285888148), 78: (40.0164190225778, 73.09405327393605), 79: (40.00746381136143, 73.01114326376296), 80: (40.0160147404569, 73.04885761186809), 81: (40.00516673665536, 73.17011662826168), 82: (40.0107513969309, 73.16353255732714), 83: (40.001884363095556, 73.01922067381473), 84: (40.00126149766185, 73.0469682323647), 85: (40.001061440229236, 73.03557052212169), 86: (40.008454978386425, 73.11330987327376), 87: (40.01235802058213, 73.14234022279167), 88: (40.00106072325966, 73.07506563046104), 89: (40.01060263556163, 73.05651980842026), 90: (40.01196155548323, 73.057404320198), 91: (40.01407058250975, 73.08720375033248), 92: (40.012947318756446, 73.09671266082393), 93: (40.011564390320096, 73.11205367891631), 94: (40.01361463778444, 73.1392344425453), 95: (40.00672859498633, 73.03511075904981), 96: (40.00677855561132, 73.09229208767411), 97: (40.00088245065831, 73.10138726977291), 98: (40.00201840518643, 73.00551840023401), 99: (40.01513060379611, 73.1174229474312)}

#epicentre to be selected from here
epicentre_coordinates = {0: (40.004810629791535, 73.0591122750594), 1: (40.01573640937946, 73.11544549706684)}

#base station location fixed
BC = (40.005467843713895, 73.02429831524648)

pdr_high = []
pdr_med = []
pdr_low = []

#iterate = input("no. of shocks : ")
iterate = 3
PDR = []
EE = []
LAT = []

# Number of fog nodes
eG = 10

# Number of mobile nodes
mG = 40

# base station event data tracker
eT = -1

# Visualization purposes
Tracker = []

# List of activeness of mobile devices
active_list = {}

T = 0
tm = 0
frequencyEvent = 2
globalEventCounter = 0

# porimary energy for mobile and fog device
energy_fog = 1000
energy_mobile = 500

# Unit for device memory
recBufferCapacity = 100
recBufferCapacity_mobile = 70
recBufferCapacity_fog = 140

# Simulation Duration
Duration =60

# Pause time
PT = 5

# Fog sensing range
sensing_range = 500.0

# Mobile sensing range
sensing_range_mobile = 200.0

# Simulation range
Xlim = [40.0, 40.0175]
Ylim = [73.0, 73.175]

minimumWaitingTime = 1

# Location of Base station
#BC = location()

# Miles to meters
mTOm = 1609.34

# Promptness increment/decrement
prompt_incentive = 5

# Distance between two latitude and longitudes
lat_dist = 69.0
lon_dist = 54.6

# Correlation between the proximity and score
Correlate = []

# How many events
how_many_events = 10


# How often should event stay in system
recur = 2

#base level energy of the node's
baseE = 10.0

# Sense event data energy
#senseE = 0.05
senseE = 0.05

# Scan event data energy
scanE = 3.68
#scanE = 0.0

# Peer to FOG data transfer energy
pr_fg_E = 0.137
#pr_fg_E = 0.0

# Peer to FOG data transfer energy
fg_fg_E = 0.37
#fg_fg_E = 0.0

# energy drained
energy_file = []

# V1 = [0 for u in range(eG)]
# V2 = [[] for u in range(eG)]

V1 = []
V2 = []

# Total Generated
Tot = []
Rec = []

# TTL assigned to events according to their priority
high_TTL = 12
med_TTL = 6
low_TTL = 3

#no of event generated by its type
no_high_events = 0
no_med_events = 0
no_low_events = 0

# Create Simpy environment and assign nodes to it.  
env = simpy.Environment()
entities = []

for i in range(eG + mG + 2):

        if i < eG:
            # Edge device
            entities.append(Node(env, 'E-' + str(i), Coordinates[i]))

        elif i < eG + mG:
            # Mobile device
            entities.append(Node(env, 'M-' + str(i), location() ))

        elif i == eG + mG:
            # Event generator
            entities.append(Node(env, 'EG-' + str(i), location()))

        else:
            # Base station
            entities.append(Node(env, 'BS-' + str(i), BC))

damaged_station = 0
alive_station = eG
epicentre = [0.0, 0.0]

node_distance =[]
max_node_distance = 0.0
min_node_distance = 0.0

for i in range(eG):
    node_distance.append(0)
    
for z in range(iterate):
    print("iteration : "+str(z))
    
    if z>0 :
        epicentre = epicentre_coordinates[z-1]
        print("epicentre is : "+str(epicentre))
        
        distance_from_epicentre()
        #print(node_distance)
        max_node_distance = max(node_distance)
        #print("max node distance from epicentre : ",max_node_distance)
        
        damaged_station = 0 
        alive_station = eG
        damage()
        
    print("damaged station ",damaged_station)
    print("alive station ", alive_station)    
    
    env.run(until = (z+1)* Duration)
    
    show_plot_1()
    show_plot_2()
    show_plot_3()
    
print(pdr_high)
print(pdr_med)
print(pdr_low)
check_duplicate_in_bs()
check_if_any_event_in_bs_ttl_less_zero()
print(energy_file)
#check_energy_status()
#print(entities[eG+mG+1].recBuf)
check_latency()