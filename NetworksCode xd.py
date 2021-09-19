
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

"""
This is an example of using the networkx library for the networks project for C&N
10/2/20 Updated for python 3 and networkx 2
"""
params = {
   'axes.labelsize': 30,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'figure.figsize': [20, 15]
   }

def find_rand_vert(a):
    b = random.choice(a)
   
    return b


def PPA_dist_gen(a):
    counter = [0 for i in range(len(a))]
    for i in range(len(a)):
        counter[a[i][0]] += 1 / len(a)
    sum_k = len(a)
   
    return counter, sum_k
   
   

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


def PPA_vert_chooser_test(edges_list, m):
    vertices = []
    target_vertices = []
    for i in range(len(edges_list)):
        vertices.append(edges_list[i][0])
        vertices.append(edges_list[i][1])
       
    test = 0
   
    for p in range(m):
        new_vert = random.choice(vertices)
        if test == 0:
            target_vertices.append(new_vert)
            test += 1
        else:
            while new_vert == target_vertices[-1]:
                new_vert = random.choice(vertices)
            target_vertices.append(new_vert)

   
    return target_vertices


#Exp cutoff
#def PPA_vert_chooser_test_effi(full_vert_list, m):
#Fat tail
#def PPA_vert_chooser_test_effi(full_edges_list, m):
#    vertices = full_edges_list
#    target_vertices = []
#    test = 0
#    
#    for p in range(m):
#        new_vert = random.choice(vertices)
#        if test == 0:
#            target_vertices.append(new_vert)
#            
#            remove_values_from_list(vertices, new_vert)
#            test += 1
#        else:
##            while new_vert == target_vertices[-1]:
##                new_vert = random.choice(vertices)
#            target_vertices.append(new_vert)
#
#    
#    return target_vertices
   
def PPA_vert_chooser_test_effi(full_edges_list, m):
   
    vertices = full_edges_list
    target_vertices = []    
    test = 0
    for p in range(m):
        if test == 0:
            new_vert = random.choice(vertices)
            target_vertices.append(new_vert)
            test += 1

        else:
            new_vert = random.choice(vertices)
            con_check = 100
            while con_check > 0:
               
                con_check = con_check_method(new_vert, target_vertices)
                if con_check > 0:
                    new_vert = random.choice(vertices)
            target_vertices.append(new_vert)  
    return target_vertices

def con_check_method(new_vert, target_vertices):
    val = 0
    for i in range(len(target_vertices)):
        if new_vert == target_vertices[i]:
            val += 1
    return val

def PPA_vert_chooser_other(edges_list, m):
    vertices = np.ravel(edges_list).tolist()
    target_vertices = []

    test = 0
    for p in range(m):
        if test == 0:
            new_vert = random.choice(vertices)
            target_vertices.append(new_vert)
            test += 1

        else:
            new_vert = random.choice(vertices)
            con_check = 100
            while con_check > 0:
               
                con_check = con_check_method(new_vert, target_vertices)
                if con_check > 0:
                    new_vert = random.choice(vertices)
            target_vertices.append(new_vert)  
   
    return target_vertices








def PPA_Iterate(t, m):
    G_0 = nx.complete_graph(m+1)
    t0 = m
    vert_list = list(G_0.nodes)

    full_edges_list = list(G_0.edges)
   
    edges_list_init = list(G_0.edges)
   
    k_list = [0 for i in range((m*t) + 10)]
   
    full_edges_find_list = []
    for i in range(len(edges_list_init)):
        for j in range(len(edges_list_init[i])):
            full_edges_find_list.append(edges_list_init[i][j])
            k_list[edges_list_init[i][j]] += 1

    for time in range(t):
        t0 += 1        

        target_verts = PPA_vert_chooser_test_effi(full_edges_find_list, m)
           
        for q in range(len(target_verts)):  
            full_edges_list.append((t0, target_verts[q]))
            full_edges_find_list.append(target_verts[q])
            full_edges_find_list.append(t0)  
            k_list[t0] += 1
            k_list[target_verts[q]] += 1
           
           
        vert_list.append(t0)

       
    final_edges = full_edges_list
    final_nodes = vert_list
   
       
    return G_0, list(final_edges), final_nodes, (G_0.edges.data), full_edges_list, k_list





def PPA_Iterate_graph(t, m):
    G_0 = nx.complete_graph(m+1)
    t0 = m
    vert_list = list(G_0.nodes)
    edges_list_init = list(G_0.edges)
   
    full_edges_find_list = []
    for i in range(len(edges_list_init)):
        for j in range(len(edges_list_init[i])):
            full_edges_find_list.append(edges_list_init[i][j])
   
    for time in range(t):
        t0 += 1
        new_vert = t0
       
        G_0.add_node(t0)
       
        target_verts = PPA_vert_chooser_other(list(G_0.edges), m)

        for q in range(len(target_verts)):  
            G_0.add_edge(new_vert, target_verts[q])
            vert_list.append(t0)            
            full_edges_find_list.append(target_verts[q])
            full_edges_find_list.append(t0)
       
    final_edges = G_0.edges
    final_nodes = G_0.nodes
   
   
       
    return G_0, list(final_edges), final_nodes, (G_0.edges.data), vert_list


   

def Iterate_m_empty(m):
   
    graph = nx.empty_graph(m)
   
           
    return graph


       
 
   
def n_k_list(edges_list):
    instances_of_each_vertex = []

    for i in range(len(edges_list)):
        for j in range(len(edges_list[i])):
            instances_of_each_vertex.append(int(edges_list[i][j]))
           
    each_vertex_counter = [0 for i in range(max(instances_of_each_vertex) + 1)]
    for i in instances_of_each_vertex:
        each_vertex_counter[i] += 1  
       
    n_k_list = [0 for i in range(max(each_vertex_counter)+1)]
   
    for i in range(len(each_vertex_counter)):
        n_k_list[each_vertex_counter[i]] += 1
   
    k = np.linspace(0, len(n_k_list)-1, len(n_k_list)).tolist()
   
    return k, n_k_list, instances_of_each_vertex, each_vertex_counter


def n_k_list_graph(edges_list):
    instances_of_each_vertex = []
    for i in range(len(edges_list)):
        for j in range(len(edges_list[i])):
            instances_of_each_vertex.append(int(edges_list[i][j]))


    each_vertex_counter = [0 for i in range(len(instances_of_each_vertex) + 1)]
    for i in instances_of_each_vertex:
        each_vertex_counter[i] += 1  
       
    n_k_list = [0 for i in range(max(each_vertex_counter)+1)]
    for i in range(len(each_vertex_counter)):
        n_k_list[each_vertex_counter[i]] += 1
   
    k = np.linspace(0, len(n_k_list)-1, len(n_k_list)).tolist()
   
    return k, n_k_list, instances_of_each_vertex, each_vertex_counter

       
def n_k_list_phase_3(instances_of_each_vertex):
           
    each_vertex_counter = [0 for i in range(max(instances_of_each_vertex) + 1)]
    for i in instances_of_each_vertex:
        each_vertex_counter[i] += 1  
       
    n_k_list = [0 for i in range(max(each_vertex_counter)+1)]
   
    for i in range(len(each_vertex_counter)):
        n_k_list[each_vertex_counter[i]] += 1
   
    k = np.linspace(0, len(n_k_list)-1, len(n_k_list)).tolist()
   
    return k, n_k_list, instances_of_each_vertex, each_vertex_counter

def logbin(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y

def p_k_counter(n_k_set):
    p = []
    N = np.sum(n_k_set)
   
    for i in range(len(n_k_set)):
        new_val = n_k_set[i] / N
        p.append(new_val)
       
    return p

def std_dev(a):
    T= len(a)
    val_1_p = []
    for i in range(len(a)):
        val_1_p.append(a[i]**2 / T)
    val_2 = (sum(a) / T) ** 2
    sig = np.sqrt(sum(val_1_p) - val_2)
   
    return sig



def PRA_Iterate(t, m):
    G_0 = nx.complete_graph((m)+1)
    t0 = m
    vert_list = list(G_0.nodes)
   
    full_edges_list = list(G_0.edges)
    k_list = [0 for i in range(2 * (m+t+1))]
    edges_list_init = list(G_0.edges)
   
    full_edges_find_list = []

           
    for i in range(len(edges_list_init)):
        k_list[i] += 1
        for j in range(len(edges_list_init[i])):
            full_edges_find_list.append(edges_list_init[i][j])
            k_list[j] += 1

    for time in range(t):
        t0 += 1        
        target_verts = PPA_vert_chooser_test_effi(vert_list, m)    
        for q in range(len(target_verts)):  
            full_edges_list.append((t0, target_verts[q]))    
            full_edges_find_list.append(target_verts[q])
            full_edges_find_list.append(t0)  
            k_list[t0] += 1
            k_list[target_verts[q]] += 1
                       
        vert_list.append(t0)      
    final_edges = full_edges_list
    final_nodes = vert_list
       
    return G_0, list(final_edges), final_nodes, (G_0.edges.data), k_list

def q_chooser(q):
    prob = random.uniform(0,1)
    if (1-q) > prob:
        return 0
   
    else:
        return 100

def neigh_chooser(v_0_neigh, v_0):
    new_vert = random.choice(v_0_neigh)
    return_val = 0
    for i in range(len(new_vert)):
        if new_vert[i] != v_0:
            return_val += new_vert[i]
    return return_val




def RW_vert_chooser(full_edges_list,vert_list, m, q):
    v_0 = random.choice(vert_list)
    prob_test = q_chooser(q)
    full_path = []
    times = 0
    while prob_test > 0:
           
        v_0_neigh = full_edges_list[v_0]
        chosen_neigh = neigh_chooser(v_0_neigh, v_0)
        prob_test = q_chooser(q)
        v_0 = chosen_neigh
        full_path.append(v_0)
        times += 1

#    print("full path is ", full_path, " after ", times, " times")
#    print("v_0 is ", v_0)
    return v_0

##HEREHERE CHECK if you can have multiple of same, if so it makes it a lot easier
#becuase we just need to run RW_vert_chooser m times
   



###NO PLURALS ALLOWED
def RW_vert_finder(full_edges_list,vert_list, m, q):
    tot_fin_list = []
    test = 0
    for p in range(m):
        if test == 0:
            new_vert = RW_vert_chooser(full_edges_list,vert_list, m, q)
            tot_fin_list.append(new_vert)
            test += 1

        else:
            new_vert = RW_vert_chooser(full_edges_list,vert_list, m, q)
            con_check = 100
            while con_check > 0:
                con_check = con_check_method(new_vert, tot_fin_list)
                if con_check > 0:
                    new_vert = RW_vert_chooser(full_edges_list,vert_list, m, q)
            tot_fin_list.append(new_vert)  
   
    return tot_fin_list

####PLURALS ALLOWED
#def RW_vert_finder(full_edges_list,vert_list, m, q):
#    tot_fin_list = []
#    for p in range(m):
#        new_vert = RW_vert_chooser(full_edges_list,vert_list, m, q)
#        tot_fin_list.append(new_vert)
#
#  
#    return tot_fin_list

   

def rand_walk_Iterate(t, m, q):
   
    G_0 = nx.complete_graph(m)
    t0 = m
    vert_list = list(G_0.nodes)    
    full_edges_list= [[] for i in range((t + m)+1)]
    edges_list_init = list(G_0.edges)
    full_edges_find_list = []
   
    for i in range(len(edges_list_init)):
        for j in range(len(edges_list_init[i])):
            full_edges_find_list.append(edges_list_init[i][j])
           
    for i in range(len(edges_list_init)):
        full_edges_list[edges_list_init[i][0]].append(edges_list_init[i])
        full_edges_list[edges_list_init[i][1]].append(edges_list_init[i])

    for time in range(t):
        t0 += 1        
       
        target_verts = RW_vert_finder(full_edges_list,vert_list, m, q)
#        print("target_verts is ", target_verts)
        for b in range(len(target_verts)):  
            full_edges_list[target_verts[b]].append((target_verts[b], t0))
            full_edges_list[t0].append((t0, target_verts[b]))
            full_edges_find_list.append(t0)
            full_edges_find_list.append(target_verts[b])

           
        vert_list.append(t0)

    final_edges = full_edges_list
    final_nodes = vert_list
   
    return G_0, list(final_edges), final_nodes, (G_0.edges.data), full_edges_list, full_edges_find_list

def cumesum(a):
    cume_sum = []
    tot = 0
#    tot_val = sum(a)
    for i in range(len(a)):
        cume_sum.append(tot + (a[i]) )
        tot += a[i]
    return cume_sum


def actual_PPA_deg_dis(k_set, m):
    p_k = []
   
    for i in range(len(k_set)):
        new_p_k = (2*m*(m+1)) / (k_set[i] *(k_set[i]+1) * (k_set[i] + 2) )
        p_k.append(new_p_k)
   
    return p_k

def k_s_test(a, b):
    max_diff = 0
    for i in range(len(a)):
        new_diff = np.abs(a[i] - b[i])
        if new_diff > max_diff:
            max_diff = new_diff
   
    return max_diff

def node_counter(node, all_edges):
    counter_0 = []
    time = []
    counter_0_val = 0
    for i in range(len(all_edges)):
        if all_edges[i][0] == 0 or all_edges[i][1] == node:
            counter_0_val += 1
            counter_0.append(counter_0_val)
            time.append(i)
        else:
            time.append(i)
            counter_0.append(counter_0_val)
       
    return time, counter_0

def find_mean(a):
    running_val = 0
    running_set = 0
    for i in range(len(a)):
        if a[i] > 0:
            running_val += a[i]
            running_set += 1
    mean = running_val / running_set
    return mean

def logbin_error(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot

    return x,y


#%%
###### PPA I.E. Phase 2    

#Showing that <k> is approximately 2m in time
time_vals = np.linspace(0, 1000, 501)

repeat_val = 100
k_avg_set = [[] for i in range(len(time_vals))]


for i in range(len(time_vals)):
    for j in range(repeat_val):
        k_avg = find_mean(PPA_Iterate(int(time_vals[i]), 2)[5])
        k_avg_set[i].append(k_avg)
        
#%%
        
plt_avg_set= []
std_dev_avg_k_val = []
for i in range(len(k_avg_set)):
    new_mean = np.mean(k_avg_set[i])
    new_err = np.std(k_avg_set[i]) / np.sqrt(repeat_val)
    plt_avg_set.append(new_mean)
    std_dev_avg_k_val.append(new_err)


fig = plt.figure(figsize=(7,5))

plt.rcParams.update({'font.size': 12})
plt.xlabel('Time (t)', fontsize = 20)
plt.ylabel('$\\langle k \\rangle$', fontsize = 20)
plt.plot(time_vals, plt_avg_set, 'o', ms = 3,label = 'Measured values of $\\langle k \\rangle$')
plt.errorbar(time_vals, plt_avg_set, yerr = std_dev_avg_k_val, xerr = 0, fmt = 'o')


plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 0.8)

#%%
#Testing for 1 graph
added_nodes = 100000
m = 2

graph_1 = PPA_Iterate(added_nodes, m)




#%%
#Plotting p(k) by k

k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))


prob_k = p_k_counter(n_k_set_2)

x_vals, prob_k_2 = logbin(each_vertex_counter, 1.2)

#%%



plt.figure()
plt.loglog(k_2, prob_k, 'o', ms = 1)
plt.loglog(x_vals, prob_k_2)

slice_fit, slice_cov = np.polyfit(np.log(k_2[m:30]), np.log(prob_k[m:30]), 1, cov =True)

x_slice = np.linspace(1, max(k_2), 1000)
y_slice= []
for i in range(len(x_slice)):
    new_val = (x_slice[i]**slice_fit[0]) * np.exp(slice_fit[1])
    y_slice.append(new_val)




#%%
#Log binning

k_prop_log_2, n_k_log_2 = logbin(each_vertex_counter, 1.2)

sum(each_vertex_counter)
#%%


####PPA

plt.loglog((k_prop_log_2), (n_k_log_2),'o',ms = 5)

PPA_prop_dist = actual_PPA_deg_dis(k_2[m:], m)
plt.loglog(k_2[m:], PPA_prop_dist, label = 'Analytic Distribution')



plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)

#%%
#Showing Each Graph grows as t^1/2

all_edges = graph_1[1]
node_counter_set = [10,100,1000, 10000]
node_labels = ['Node 10', 'Node 100', 'Node 1000', 'Node 10000']
full_counter = []
time_set = []

for i in node_counter_set:
    time, counter_0 = node_counter(i, all_edges)
    full_counter.append(counter_0)
    time_set.append(time)

plt.xlabel("Time")
plt.ylabel("Degrees k")

for i in range(len(full_counter)):
   
    plt.loglog(time_set[i], full_counter[i], label = node_labels[i])

plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 1)



#%%
#Changing m and examining how it changes
#Combining sets of produced values from repeats

M_set = [2, 4, 8, 16, 32, 64, 128]

k_prop_set = []
n_k_set_full = []
each_vert_counter_full = []
repeat_set = [500,500,500,500,500,200,100]
N = 10000


Phase_2_Question_3_k_prop_set = [[] for i in range(len(M_set))]
Phase_2_Question_3_n_k_set_full = [[] for i in range(len(M_set))]
Phase_2_Question_3_each_vert_counter_full = [[] for i in range(len(M_set))]
k_max_3 = [[] for i in range(len(M_set))]

n = 0
   
for q in range(len(M_set)):

      
    for d in range(repeat_set[q]):

        graph_1 = PPA_Iterate(N, M_set[q])

        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))

        for i in range(len(k_2)):
            Phase_2_Question_3_k_prop_set[q].append(k_2[i])  
        for i in range(len(n_k_set_2)):
            Phase_2_Question_3_n_k_set_full[q].append(n_k_set_2[i])

        for i in range(len(each_vertex_counter)):
            Phase_2_Question_3_each_vert_counter_full[q].append(each_vertex_counter[i])

        print("d ", d, " done")

    n+=1

    print(q, ' done')

#%%
#Producing theoretical distributions

m_labels = ['m = 2', 'm = 4','m = 8','m = 16','m = 32','m = 64','m = 128']

color_set= ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

k_val_full = np.linspace(0.5, 4500, 1000)
prob_theo_p_1_3 = [[] for i in range(len(M_set))]
for i in range(len(M_set)):
    for j in range(len(k_val_full)):
            new_val = 2*M_set[i]*(M_set[i]+1)/(k_val_full[j] * (k_val_full[j]+1)*(k_val_full[j]+2))
            prob_theo_p_1_3[i].append(new_val)
            
#%%
#Finding errors on the log-binned data 

   
error_set = [[] for i in range(len(M_set))]
error_set_x = [[] for i in range(len(M_set))]
         
for q in range(len(M_set)):
      
    
    k_log_bin_err_set = []
    p_vals_bin_err_set = []
    
    for d in range(repeat_set[q]):
        graph_1 = PPA_Iterate(N, M_set[q])
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
        k_log_bin_err, p_vals_bin_err = logbin_error(each_vertex_counter, 1.2)
        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())
    
    max_set_val = 0
    max_i = 0
    for i in range(len(k_log_bin_err_set)):
        if len(k_log_bin_err_set[i]) > max_set_val:
            max_i = i
            max_set_val = len(k_log_bin_err_set[i])
    error_set_x[q] = k_log_bin_err_set[max_i]
    
    full_p_val_set = [[] for i in range(max_set_val)]
    for i in range(len(p_vals_bin_err_set)):
        for j in range(len(p_vals_bin_err_set[i])):
            new_p_val = p_vals_bin_err_set[i][j]
            full_p_val_set[j].append(new_p_val)
            
    temp_err_set = []
    
    for i in range(len(full_p_val_set)):
        new_err = np.std(full_p_val_set[i]) / np.sqrt(len(full_p_val_set[i]))
        temp_err_set.append(new_err)
    
    error_set[q] = temp_err_set
        
    print(q, ' done')            
            
#%%
#Plotting log-binned graphs wih error bars
#2.3.2 Visualisation for numerical and theoretical values


fig = plt.figure(figsize=(6,5))

fit_vals = []
cov_vals = []

for i in range(len(Phase_2_Question_3_each_vert_counter_full)):

    k_prop_log, n_k_log = logbin(Phase_2_Question_3_each_vert_counter_full[i], 1.2)
    
    plt.xlabel("k", fontsize = 20)
    plt.ylabel("P(k,N)", fontsize = 20)
    plt.loglog(k_val_full, prob_theo_p_1_3[i], '--', color = color_set[i])
    temp_error_set = [0 for i in range(len(k_prop_log))]
    
    for c in range(len(k_prop_log)):
        for j in range(len(error_set[i])):
            if k_prop_log[c] == error_set_x[i][j]:
                temp_error_set[c] = error_set[i][j]
                

    plt.loglog(k_prop_log, n_k_log,ms = 3,color = color_set[i])
    plt.errorbar(k_prop_log, n_k_log, yerr = np.array(temp_error_set), xerr = 0, fmt = 'o', color = color_set[i], label = m_labels[i])
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)
plt.show()




#%%
#For individual m
#For cumulative distribution

m_set_test_val = 0

fig = plt.figure(figsize=(6,5))


k_prop_log, n_k_log = logbin(Phase_2_Question_3_each_vert_counter_full[m_set_test_val], 1.2)

plt.xlabel("k", fontsize = 20)
plt.ylabel("P(k,N)", fontsize = 20)
plt.loglog(k_val_full, prob_theo_p_1_3[m_set_test_val], '--', color = color_set[m_set_test_val])
   
plt.loglog(k_prop_log, n_k_log,'o',ms = 4,color = color_set[m_set_test_val],label = m_labels[m_set_test_val])

   
   
plt.xlim([1, 5000])
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)




#%%
#2.3.3 statistics

from scipy import stats

m_val = M_set[m_set_test_val]

prob_theo_1_cum_fre= []
prob_theo_1_cum_fre_chi = []

k_cum_sum_val = k_prop_log


for j in range(len(k_cum_sum_val)):
    new_val = 2*m_val*(m_val+1)/(k_cum_sum_val[j] * (k_cum_sum_val[j]+1)*(k_cum_sum_val[j]+2))
    prob_theo_1_cum_fre.append(new_val)

#%%
#Generating data for comparison test

N = 100000
n_k_log_original_num = []
theo_original_num = []
for i in range(len(n_k_log)):
    new_val = n_k_log.tolist()[i] * N
    n_k_log_original_num.append(new_val)
    
for i in range(len(prob_theo_1_cum_fre)):
    theo_val = prob_theo_1_cum_fre[i] * N
    theo_original_num.append(theo_val)

#%%
#Visual check 
plt.loglog(k_prop_log, n_k_log_original_num)
plt.loglog(k_cum_sum_val, theo_original_num)


#%%
g = stats.ks_2samp(n_k_log_original_num,theo_original_num)
h = stats.chisquare(n_k_log_original_num, theo_original_num)

print('ks test results are ', g)
print('chi square test results are ', h, ' not will not be used for PRA') # <--- very bad results as expected

#%%

a = cumesum(n_k_log.tolist())
b = cumesum(prob_theo_1_cum_fre)

#%%
#Cumulative distribution plots, with zoomed in region


fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots() 
ax.step(np.log(k_prop_log), np.log(b),label = 'Theoretical Cumulative Prob', color = 'orange')
ax.step(np.log(k_prop_log), np.log(a), drawstyle='steps-pre',linestyle='--', label = 'Numerical Cumulative Prob.', color = 'blue')

plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1, loc = 'center right')

plt.xlabel('ln(k)')
plt.ylabel('ln(CDF)')

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(ax, 1600, loc=4)
axins.xaxis.tick_top()


x1, x2, y1, y2 = 0.840,0.843 , -0.35663, -0.35675

axins.set_xlim(x1, x2) 
axins.set_ylim(y1, y2)

plt.yticks(visible=True, fontsize = 10)
plt.xticks(visible=True, fontsize = 10)


axins.step(np.log(k_prop_log), np.log(b), color = 'orange')
axins.step(np.log(k_prop_log), np.log(a),drawstyle='steps-pre',linestyle= '--',  color = 'blue')



plt.show()



#%%
#Manually finding the max-differnece between cum. dis. functions
max_diff = 0
for i in range(len(a)):
    new_val = np.abs(a[i] - b[i])
    if new_val > max_diff:
        max_diff = new_val
        print("here for i = ", i)



#%%
#Phase 2 Question 4
#Investigation into changing N
N_set = [100, 1000, 10000, 100000, 1000000]

N_set_label = [ 'N = 100', 'N = 1000', 'N = 10000', 'N = 100000', 'N = 1000000']
repeat_set = [ 1000000, 100000, 10000,1000, 100]


m_val = 2

Phase_2_Question_4_k_prop_set = [[] for i in range(len(N_set))]
Phase_2_Question_4_n_k_set_full = [[] for i in range(len(N_set))]
Phase_2_Question_4_each_vert_counter_full = [[] for i in range(len(N_set))]




   
for q in range(len(N_set)):
       


    for i in range(repeat_set[q]):
        graph_1 = PPA_Iterate(N_set[q], m_val)
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
       
        for i in range(len(k_2)):
            Phase_2_Question_4_k_prop_set[q].append(k_2[i])  
        for i in range(len(n_k_set_2)):
            Phase_2_Question_4_n_k_set_full[q].append(n_k_set_2[i])
        for i in range(len(each_vertex_counter)):
            Phase_2_Question_4_each_vert_counter_full[q].append(each_vertex_counter[i])

    print(q, ' done')
                   

#%%
#Different N errors 
     
repeat_set = [1000000, 100000, 20000, 2000, 500]
   
error_set_N = [[] for i in range(len(N_set))]
error_set_x_N = [[] for i in range(len(N_set))]
 
for q in range(len(N_set)):

    
    
    k_log_bin_err_set = []
    p_vals_bin_err_set = []
    
    for d in range(repeat_set[q]):
        graph_1 = PPA_Iterate(N_set[q], 2)
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
        k_log_bin_err, p_vals_bin_err = logbin_error(each_vertex_counter, 1.2)
        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())

    
    max_set_val = 0
    max_i = 0
    for i in range(len(k_log_bin_err_set)):
        if len(k_log_bin_err_set[i]) > max_set_val:
            max_i = i
            max_set_val = len(k_log_bin_err_set[i])
    error_set_x_N[q] = k_log_bin_err_set[max_i]
    
    full_p_val_set = [[] for i in range(max_set_val)]
    for i in range(len(p_vals_bin_err_set)):
        for j in range(len(p_vals_bin_err_set[i])):
            new_p_val = p_vals_bin_err_set[i][j]
            full_p_val_set[j].append(new_p_val)
            
    temp_err_set = []
    
    for i in range(len(full_p_val_set)):
        new_err = np.std(full_p_val_set[i]) / np.sqrt(len(full_p_val_set[i]))
        temp_err_set.append(new_err)
    
    error_set_N[q] = temp_err_set
        

    print(q, ' done')  
    
    
#%%

k_val_full = np.linspace(1.5, 10000, 10001)
prob_theo = []
for j in range(len(k_val_full)):
        new_val = 2*m_val*(m_val+1)/(k_val_full[j] * (k_val_full[j]+1)*(k_val_full[j]+2))
        prob_theo.append(new_val)
plt.rcParams.update({'font.size': 16})
#%%
fig = plt.figure(figsize=(6,5))

plt.rcParams.update({'font.size': 16})


plt.loglog(k_val_full, prob_theo, '--', label = '$p_{\\infty k}$')

for i in range(len(Phase_2_Question_4_each_vert_counter_full)):

    k_prop_log, n_k_log = logbin(Phase_2_Question_4_each_vert_counter_full[i], 1.2)


    plt.xlabel("k", fontsize = 35)
    plt.ylabel("P(k,m)", fontsize = 35)
   
    
    
    temp_error_set = [0 for i in range(len(k_prop_log))]        
    for c in range(len(k_prop_log)):
        for j in range(len(error_set_N[i])):
            if k_prop_log[c] == error_set_x_N[i][j]:
                temp_error_set[c] = error_set_N[i][j]  
                
    plt.errorbar(k_prop_log, n_k_log, yerr = np.array(temp_error_set), xerr = 0, ms = 1, fmt = 'o', capsize = 3,label = N_set_label[i], color = color_set[i])

    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 14, markerscale = 1, loc = 'upper right')
    
plt.show()

#%%
#Theoretical k1 values

k_1_set = []

for i in range(len(N_set)):
    new_val = 0.5 * (-1 + np.sqrt(1+4*(N_set[i] * m_val * (m_val+1))))
    k_1_set.append(new_val)




#%%
#Data collapsing vertical

plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(6,5))

for p in range(len(Phase_2_Question_4_each_vert_counter_full)):

    k_prop_log, n_k_log = logbin(Phase_2_Question_4_each_vert_counter_full[p], 1.2)


    k_s = []
    for j in range(len(k_prop_log)):
        new_val = 2*m_val*(m_val+1)/(k_prop_log[j] * (k_prop_log[j]+1)*(k_prop_log[j]+2))
        k_s.append(new_val)
        
    n_k_log_times_k = []
    for i in range(len(n_k_log)):
        new_val = n_k_log[i] * (1/k_s[i])
        n_k_log_times_k.append(new_val)
      
    k_prop_log_over_N = []
    for i in range(len(k_prop_log)):
        new_val = k_prop_log[i] / (k_1_set[p])
        k_prop_log_over_N.append(new_val)
            
    temp_error_set = [0 for i in range(len(k_prop_log))]

    for c in range(len(k_prop_log)):
        for j in range(len(error_set_x_N[p])):
            if k_prop_log[c] == error_set_x_N[p][j]:
                mult_val = 2*m_val*(m_val+1)/(k_prop_log[c] * (k_prop_log[c]+1)*(k_prop_log[c]+2))
                new_err = error_set_N[p][j] * (1/k_s[c])
                temp_error_set[c] = new_err
       
        
    plt.xlabel("k", fontsize = 30)
    plt.ylabel("$k ^ {\\tau_{s}}$P(k,N)", fontsize = 30)
   
    plt.loglog(k_prop_log, n_k_log_times_k,'o', ms = 2)

    plt.errorbar(k_prop_log, n_k_log_times_k, xerr= 0, yerr= temp_error_set, ms = 1, fmt = 'o', capsize = 3, label = N_set_label[p], color = color_set[p])

    
    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)

plt.show()

#%%
#Data collapsing Full

#plt.figure()
plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(6,5))

for p in range(len(N_set)):

    k_prop_log, n_k_log = logbin(Phase_2_Question_4_each_vert_counter_full[p], 1.2)

    k_s = []
    for j in range(len(k_prop_log)):
        new_val = 2*m_val*(m_val+1)/(k_prop_log[j] * (k_prop_log[j]+1)*(k_prop_log[j]+2))
        k_s.append(new_val)
       
    n_k_log_times_k = []
    for i in range(len(n_k_log)):
        new_val = n_k_log[i] * (1/k_s[i])
        n_k_log_times_k.append(new_val)
       
    temp_error_set = [0 for i in range(len(k_prop_log))]

    for c in range(len(k_prop_log)):
        for j in range(len(error_set_x_N[p])):
            if k_prop_log[c] == error_set_x_N[p][j]:
                mult_val = 2*m_val*(m_val+1)/(k_prop_log[c] * (k_prop_log[c]+1)*(k_prop_log[c]+2))
                new_err = error_set_N[p][j] / mult_val
                temp_error_set[c] = new_err
    
    
    k_prop_log_over_N = []
    for i in range(len(k_prop_log)):
        new_val = k_prop_log[i] / (k_1_set[p])
        k_prop_log_over_N.append(new_val)
        
    plt.xlabel("k / $k_{1}$", fontsize = 30)
    plt.ylabel("P(k,N) / $P_{k \\infty}$", fontsize = 30)
   
    plt.loglog(k_prop_log_over_N, n_k_log_times_k, ms = 2, label = N_set_label[p])



    
    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)
plt.show()

#%%
#Data collapsing to Gaussian form. 
#limit = int(max(N_set) ** 0.5) * 10

#Because we just want the maximum values, we can run this independently
#to get more repeats at a much better rate (less date to process)

N_set = [100, 1000, 10000, 100000, 1000000]


k_max = [[] for i in range(len(N_set))]
repeat_vals = [10000, 10000, 10000, 5000, 2000]
   
test_val = []

for q in range(len(N_set)):


    for i in range(repeat_vals[q]):

        
        k = PPA_Iterate(N_set[q], 2)[5]
        k_max[q].append(max(k))




    print(q, ' done')
       
#%%
#Calculating k_max for each value
    
average_k_max_set = []
k_max_set_errs = []
for i in range(len(k_max)):
    new_val = np.mean(k_max[i])
    new_err = np.std(k_max[i]) / np.sqrt(len(k_max[i]))
    print("std is ", np.std(k_max[i]))
    print("length is ", len(k_max[i]))
    print("error is ", new_err)
    average_k_max_set.append(new_val)
    k_max_set_errs.append(new_err)
    
#Theoretical values of k_max

theo_k_max = []

for i in range(len(k_max)):
    new_val = (0.5) * (1 + np.sqrt(4 * N_set[i] * 2 * (2 + 1)))
    theo_k_max.append(new_val)
    
#%%
#Plotting kmax against k1
    
    
fig = plt.figure(figsize=(6,5))

plt.rcParams.update({'font.size': 16})
plt.xlabel('N', fontsize = 20)
plt.ylabel('$k_{1}$', fontsize = 20)

k_max_set_errs_zoom = []

for i in range(len(k_max_set_errs)):
    new_val = k_max_set_errs[i] * 50
    k_max_set_errs_zoom.append(new_val)

plt.loglog(N_set, theo_k_max, '--', label = 'Theoretical $k_{1}$')

plt.errorbar(N_set, average_k_max_set, yerr = k_max_set_errs_zoom, xerr = 0, ms = 4,capsize = 3, fmt = 'o', label = '$k_{max}$')


k_max_fit, k_max_cov = np.polyfit(np.log(N_set), np.log(average_k_max_set), 1, cov = True)

k_1_x = np.linspace(100, 10**6, 1000)
k_1_y = []

for i in range(len(k_1_x)):
    new_val = (k_1_x[i] ** k_max_fit[0]) * np.exp(k_max_fit[1])
    k_1_y.append(new_val)
    
plt.loglog(k_1_x, k_1_y, '--', label = '$k_{max}$ Best Fit Line')


plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 0.5)
plt.savefig('test k max.png', dpi=1600)

#%%
#Calculating kmax/k1

k_1_y_divided = []

for i in range(len(average_k_max_set)):
    new_val = average_k_max_set[i] / theo_k_max[i]
    k_1_y_divided.append(new_val)

k_1_div_err_set = []
for i in range(len(k_max_set_errs_zoom)):
    new_err = k_max_set_errs_zoom[i] / (theo_k_max[i] * 50)
    k_1_div_err_set.append(new_err)

k_div_fit, k_div_cov = np.polyfit(N_set, k_1_y_divided, 1, cov = True)
x_div = np.linspace(min(N_set) - 0.1, max(N_set), 1000)
y_div = []
for i in range(len(x_div)):
    new_val = k_div_fit[1] 
    y_div.append(new_val)

#%%
#Plotting kmax/k1 against N
fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

plt.semilogx(N_set, k_1_y_divided, 'o', label = '$k_{max} / k_{1}$', color = 'blue')

plt.semilogx(x_div, y_div, '--', color = 'orange',label = 'Best Fit Line')
plt.xlabel("N", fontsize = 20)
plt.ylabel("$k_{1} / \\alpha N^{\\beta}$", fontsize = 20,)
plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)


plt.errorbar(N_set, k_1_y_divided, yerr = k_1_div_err_set, xerr = 0, ms = 3,capsize = 3, fmt = 'o',color = color_set[0], label = 'Measured $k_{max}$')

plt.savefig("divided graph", dpi = 1600)




#%%
#Preparing Gaussian distributions

N_set_label = ['N = 100', 'N = 1000', 'N = 10000']
limit = 0


limit = 10000
prob_vals_y_first = [[0 for i in range(limit)] for j in range(len(k_max))]
prob_vals_x = [[0 for i in range(limit)] for j in range(len(k_max))]

for i in range(len(k_max)):
    for j in range((len(k_max[i]))):
        norm_factor = sum(k_max[i])
        new_val = int(k_max[i][j])
        prob_vals_y_first[i][new_val] += 1
       
       
prob_vals_y  = [[] for j in range(len(k_max))]  

for i in range(len(prob_vals_y_first)):
    for j in range(len(prob_vals_y_first[i])):
        new_val = prob_vals_y_first[i][j] / sum(prob_vals_y_first[i])
        prob_vals_y[i].append(new_val)
       
#%%
#Plotting Gaussian distributions for N above
        
fig = plt.figure(figsize=(5,5))
color_set_alt = ['blue', 'orange', 'green']
plt.figure()
for i in range(len(prob_vals_y)):
    plt.plot(prob_vals_y[2-i], label = N_set_label[2-i])
plt.xlim(0, 450)
plt.xlabel("k", fontsize = 20)
plt.ylabel("P(N,k)", fontsize = 20)
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)
plt.rcParams.update({'font.size': 16})

#plt.savefig("gaussian for ppa 1", dpi = 1200)

#%%
#Getting the standard deviations and means and data collapsing

Gauss_vals_PPA = [[] for i in range(len(k_max))]
Gauss_x_PPA = [[] for i in range(len(k_max))]


sigma_set =[]
mean_set = []

for i in range(len(Gauss_vals_PPA)):
    for j in range(len(prob_vals_y[i])):
        if prob_vals_y[i][j] > 0:
            Gauss_vals_PPA[i].append(prob_vals_y[i][j])
            Gauss_x_PPA[i].append(j)
           
           
           

for i in range(len(k_max)):
    sigma = std_dev(k_max[i])
    sigma_set.append(sigma)
    mean = np.mean(k_max[i])
    mean_set.append(mean)
   
new_y_Gauss_val = [[] for i in range(len(Gauss_vals_PPA))]
for i in range(len(Gauss_vals_PPA)):
    for j in range(len(Gauss_vals_PPA[i])):
        new_val = Gauss_vals_PPA[i][j] * sigma_set[i] * np.sqrt(2 * np.pi)
        new_y_Gauss_val[i].append(new_val)
       
new_x_Gauss_val = [[] for i in range(len(Gauss_vals_PPA))]

for i in range(len(Gauss_vals_PPA)):
    for j in range(len(Gauss_vals_PPA[i])):
        new_val = (Gauss_x_PPA[i][j] - mean_set[i])/sigma_set[i]
        new_x_Gauss_val[i].append(new_val)

fig = plt.figure(figsize=(6,5))
plt.xlabel("(k - $\\langle k \\rangle$) / $\\sigma_{k}$", fontsize = 20)
plt.ylabel("$\\sigma_{k}$P(k;N)", fontsize = 20)   
for i in range(len(k_max)):
    plt.plot(new_x_Gauss_val[2-i], new_y_Gauss_val[2-i],'o', ms = 2, label = N_set_label[2-i])

import math
plt.rcParams.update({'font.size': 16})

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-4, 4, 1000)
plt.plot(x, np.sqrt(2 * np.pi ) * stats.norm.pdf(x, mu, sigma), '--', label = 'Gaussian Dist.')

   
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 2)




#%%

#PHASE 3: Pure Random Attatchment


m = 2
added_nodes = 100000


graph_1 = PRA_Iterate(added_nodes, m)





#%%
#Plotting n(k) by k

k_3, n_k_set_3, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))

prob_k = p_k_counter(n_k_set_3)

#%%
#Log binning

k_prop_log_3, n_k_log_3 = logbin(each_vertex_counter, 1)


#%%
#Visual check

plt.loglog(k_3, prob_k)
plt.loglog(k_prop_log_3, n_k_log_3, '--')
plt.xlabel('k')
plt.ylabel('p(k;N)')




#%%
#Phase 3 Question 4
M_set = [2,4,8,16,32,64,128]

m_labels = ['m = 2', 'm = 4', 'm = 8', 'm = 16', 'm = 32', 'm = 64', 'm = 128']
repeat_set = [1000, 1000, 1000, 1000, 1000, 500, 100]
repeat_val = 10
N = 100000
num = 2
Phase_3_Question_3_k_prop_set = [[] for i in range(len(M_set))]
Phase_3_Question_3_n_k_set_full = [[] for i in range(len(M_set))]
Phase_3_Question_3_each_vert_counter_full = [[] for i in range(len(M_set))]

k_max_3 = [[] for i in range(len(M_set))]

   
for q in range(len(M_set)):

    for p in range(repeat_set[q]):
        graph_1 = PRA_Iterate(N, M_set[q])
        k_3, n_k_set_3, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
       
        for i in range(len(k_2)):
            Phase_3_Question_3_k_prop_set[q].append(k_2[i])  
        for i in range(len(n_k_set_2)):
            Phase_3_Question_3_n_k_set_full[q].append(n_k_set_2[i])

        for i in range(len(each_vertex_counter)):
            Phase_3_Question_3_each_vert_counter_full[q].append(each_vertex_counter[i])
        k_max_3[q].append(max(graph_1[4]))
        
    print(q, ' done')
   #%% 
    #log-bin errors
rep_err_val = [500, 500, 500, 500, 100, 100, 100]
   
error_set_pra = [[] for i in range(len(M_set))]
error_set_x_pra = [[] for i in range(len(M_set))]
         
for q in range(len(M_set)):
      
    
    k_log_bin_err_set = []
    p_vals_bin_err_set = []
    
    for d in range(rep_err_val):
        graph_1 = PRA_Iterate(N, M_set[q])
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
        k_log_bin_err, p_vals_bin_err = logbin_error(each_vertex_counter, 1)
        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())
        print('d ', d , ' is done')
    
    max_set_val = 0
    max_i = 0
    for i in range(len(k_log_bin_err_set)):
        if len(k_log_bin_err_set[i]) > max_set_val:
            max_i = i
            max_set_val = len(k_log_bin_err_set[i])
    error_set_x_pra[q] = k_log_bin_err_set[max_i]
    
    full_p_val_set = [[] for i in range(max_set_val)]
    for i in range(len(p_vals_bin_err_set)):
        for j in range(len(p_vals_bin_err_set[i])):
            new_p_val = p_vals_bin_err_set[i][j]
            full_p_val_set[j].append(new_p_val)
            
    temp_err_set = []
    
    for i in range(len(full_p_val_set)):
        new_err = np.std(full_p_val_set[i]) / np.sqrt(len(full_p_val_set[i]))
        temp_err_set.append(new_err)
    
    error_set_pra[q] = temp_err_set
        

    print(q, ' done')                   
    
#%%
    
M_set = [2,4,8,16,32,64,128]
m_labels = ['m = 2', 'm = 4', 'm = 8', 'm = 16', 'm = 32', 'm = 64', 'm = 128']

Phase_3_Question_3_p_k_fit = [[] for i in range(len(M_set))]
prob_theo_3_p_3 = [[] for i in range(len(M_set))]

k_val_full_set = [[] for i in range(len(M_set))]
for i in range(len(k_val_full_set)):
    k_val_full = np.linspace(1, max(Phase_3_Question_3_k_prop_set[i]), max(Phase_3_Question_3_k_prop_set[i]))
    k_val_full_set[i] = k_val_full
    
for i in range(len(M_set)):
    for j in range(len(k_val_full_set[i])):
            new_val =  (1 / (1+M_set[i]))*((M_set[i] / (1 + M_set[i])) ** (k_val_full_set[i][j] - M_set[i]))
            prob_theo_3_p_3[i].append(float(new_val))
     
#%%
#Plotting PRA distribution with theoretical for all M
            
Phase_3_Question_3_p_k_fit = [[] for i in range(len(M_set))]
fig = plt.figure(figsize=(6,5))


for i in range(len(Phase_3_Question_3_k_prop_set)):

    
    k_prop_log, n_k_log = logbin(Phase_3_Question_3_each_vert_counter_full[i], 1.)

    plt.xlabel("k", fontsize = 20)
    plt.ylabel("P(k, N)", fontsize = 20)
    temp_error_set = [0 for i in range(len(k_prop_log))]

    for c in range(len(k_prop_log)):
        for j in range(len(error_set_pra[i])):
            if k_prop_log[c] == error_set_x_pra[i][j]:

                temp_error_set[c] = error_set_pra[i][j]


    plt.loglog(k_val_full_set[i], prob_theo_3_p_3[i], '--', color = color_set[i])
    plt.errorbar(k_prop_log, n_k_log, yerr = np.array(temp_error_set), xerr = 0, fmt = 'o', color = color_set[i], label = m_labels[i])


    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)



 


   
#%%
#Phase 3 Question 4
#Changing N (Not needed)

N_set = [ 100, 1000, 10000, 100000, 1000000]
N_set_label = ['N = 100', 'N = 1000', 'N = 10000', 'N = 100000', 'N = 1000000']
repeat_set = [ 10000, 10000, 10000,1000, 100]
repeat_level = 5



num = 2
M = 5

Phase_3_Question_4_k_prop_set = [[] for i in range(len(N_set))]
Phase_3_Question_4_n_k_set_full = [[] for i in range(len(N_set))]
Phase_3_Question_4_each_vert_counter_full = [[] for i in range(len(N_set))]
   
for q in range(len(repeat_set)):
       

    k_supa_set = []
    n_k_supa_set = []
    instances_supa = []
    vertex_counter_supa = []
    largest_k_supa = []

    for i in range(repeat_set[q]):
        graph_1 = PRA_Iterate(N_set[q], num)
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list(list(graph_1[1]))
       
        for i in range(len(k_2)):
            Phase_3_Question_4_k_prop_set[q].append(k_2[i])  
        for i in range(len(n_k_set_2)):
            Phase_3_Question_4_n_k_set_full[q].append(n_k_set_2[i])

        for i in range(len(each_vertex_counter)):
            Phase_3_Question_4_each_vert_counter_full[q].append(each_vertex_counter[i])


        largest_k_supa.append(max(each_vertex_counter))


    print(q, ' done')
       
       
#%%
#Not needed
Phase_3_Question_4_p_k_fit = [[] for i in range(len(N_set))]


plt.figure()
for i in range(len(Phase_3_Question_4_k_prop_set)):
    k_prop_log, n_k_log = logbin(Phase_3_Question_4_each_vert_counter_full[i], 1.2)


    plt.xlabel("k", fontsize = 30)
    plt.ylabel("log bin of node distribution?", fontsize = 30)
   
    plt.loglog(k_prop_log, n_k_log,ms = 2,label = N_set_label[i])

   
   
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)
plt.show()


#%%
#LARGEST DEGREE SIZES



k_max_p_3 = [[] for i in range(len(N_set))]

repeat_set = [10000, 10000, 10000, 5000, 1000]
   
for q in range(len(N_set)):

   
    for i in range(repeat_set[q]):
        k = PRA_Iterate(N_set[q], num)[4]
        k_max_p_3[q].append(max(k))


    print(q, ' done')
   
#%%  
#Plotting k1 and kmax against N


fig = plt.figure(figsize=(6,5))

plt.rcParams.update({'font.size': 14})

average_k_3_max_set = []
k_max_3_set_errs = []
for i in range(len(k_max_p_3)):
    new_val = np.mean(k_max_p_3[i])
    new_err = np.std(k_max_p_3[i]) / np.sqrt(len(k_max_p_3[i]))
    average_k_3_max_set.append(new_val)
    k_max_3_set_errs.append(new_err)
    
#Theoretical values of k_max

theo_k_3_max = []
x_vals_theo = np.linspace(min(N_set), max(N_set), 1000000)

for i in range(len(x_vals_theo)):
    new_val = num - (np.log(x_vals_theo[i]) / (np.log(num) - np.log(num+1)))
    theo_k_3_max.append(new_val)
    
plt.xlabel('N', fontsize = 20)
plt.ylabel('$k_{1}$', fontsize = 20)


plt.loglog(x_vals_theo, theo_k_3_max, '--', label = 'Theoretical $k_{1}$')
plt.errorbar(N_set, average_k_3_max_set, yerr = k_max_3_set_errs, xerr = 0, ms = 4,capsize = 3, fmt = 'o', label = 'Measured $k_{1}$')

k_max_3_fit, k_max_3_cov = np.polyfit(np.log(N_set), np.log(average_k_3_max_set), 1, cov = True)

plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 0.8)




theo_k_3_comparison = []

for i in range(len(N_set)):
    new_val = num - (np.log(N_set[i]) / (np.log(num) - np.log(num+1)))
    theo_k_3_comparison.append(new_val)

#%%
#Dividing kmax by k1

k_1_y_divided = []

for i in range(len(average_k_3_max_set)):
    new_val = average_k_3_max_set[i] / theo_k_3_comparison[i]
    k_1_y_divided.append(new_val)


k_div_fit, k_div_cov = np.polyfit(N_set, k_1_y_divided, 1, cov = True)
x_div = np.linspace(min(N_set) - 0.1, max(N_set), 1000)
y_div = []
for i in range(len(x_div)):
    new_val = k_div_fit[1]
    y_div.append(new_val)

#%%
fig = plt.figure(figsize=(6,5))

plt.semilogx(N_set, k_1_y_divided, 'o')

k_1_y_divided_pra_errors = []

for i in range(len(k_max_3_set_errs)):
    new_error = k_max_3_set_errs[i] / theo_k_3_comparison[i]
    k_1_y_divided_pra_errors.append(new_error)
    
    
plt.errorbar(N_set, k_1_y_divided, yerr = k_1_y_divided_pra_errors, xerr = 0, ms = 4,capsize = 3, fmt = 'o', label = '$k_{max} / k_{1}$')


plt.xlabel("N", fontsize = 20)
plt.ylabel("$k_{max} / k_{1}$", fontsize = 20)
plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 17, markerscale = 1)





#%%
#Full Gaussian probability distributions 
fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})


plt.xlabel('k', fontsize = 20)
plt.ylabel('p(k, N)', fontsize = 20)

plt.xlim(6, 47)
plt.ylim(0, 0.3)

limit = 0
for i in range(len(k_max_p_3)):
    for j in range(len(k_max_p_3[i])):
        if k_max_p_3[i][j] > limit:
            limit = k_max_p_3[i][j] + 1

prob_vals_y_first_p_3 = [[0 for i in range(int(limit))] for j in range(len(k_max_p_3))]
prob_vals_x_p_3 = [[0 for i in range(int(limit))] for j in range(len(k_max_p_3))]

for i in range(len(k_max_p_3)):
    for j in range(len(k_max_p_3[i])):
        norm_factor = sum(k_max_p_3[i])
        new_val = int(k_max_p_3[i][j])

        prob_vals_y_first_p_3[i][new_val] += 1
       
       
prob_vals_y  = [[] for j in range(len(k_max_p_3))]  

for i in range(len(prob_vals_y_first_p_3)):
    for j in range(len(prob_vals_y_first_p_3[i])):
        new_val = prob_vals_y_first_p_3[i][j] / sum(prob_vals_y_first_p_3[i])
        prob_vals_y[i].append(new_val)
       
for i in range(len(prob_vals_y)):
    plt.plot(prob_vals_y[i], label = N_set_label[i])
   
plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 14, markerscale = 1)


#%%
#Getting the standard deviations and means and data collapsing the Gaussian forms

fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

Gauss_vals_PPA_p_3 = [[] for i in range(len(k_max_p_3))]
Gauss_x_PPA_p_3 = [[] for i in range(len(k_max_p_3))]


sigma_set =[]
mean_set = []

for i in range(len(Gauss_vals_PPA_p_3)):
    for j in range(len(prob_vals_y[i])):
        if prob_vals_y[i][j] > 0:
            Gauss_vals_PPA_p_3[i].append(prob_vals_y[i][j])
            Gauss_x_PPA_p_3[i].append(j)
           
           
           

for i in range(len(k_max_p_3)):
    sigma = std_dev(k_max_p_3[i])
    sigma_set.append(sigma)
    mean = np.mean(k_max_p_3[i])
    mean_set.append(mean)
   
new_y_Gauss_val = [[] for i in range(len(Gauss_vals_PPA_p_3))]
for i in range(len(Gauss_vals_PPA_p_3)):
    for j in range(len(Gauss_vals_PPA_p_3[i])):
        new_val = Gauss_vals_PPA_p_3[i][j] * sigma_set[i] * np.sqrt(2 * np.pi)
        new_y_Gauss_val[i].append(new_val)
       
new_x_Gauss_val = [[] for i in range(len(Gauss_vals_PPA_p_3))]

for i in range(len(Gauss_vals_PPA_p_3)):
    for j in range(len(Gauss_vals_PPA_p_3[i])):
        new_val = (Gauss_x_PPA_p_3[i][j] - mean_set[i])/sigma_set[i]
        new_x_Gauss_val[i].append(new_val)

       
plt.xlabel("(k - $\\langle k \\rangle$) / $\\sigma_{k}$", fontsize = 20)
plt.ylabel("$\\sigma_{k}$P(k;N)", fontsize = 20) 

for i in range(len(new_x_Gauss_val)):
    plt.plot(new_x_Gauss_val[i], new_y_Gauss_val[i],'o',ms = 5, label = N_set_label[i])

 
import math
from scipy import stats

plt.rcParams.update({'font.size': 16})

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-4, 4, 1000)
plt.plot(x, np.sqrt(2 * np.pi ) * stats.norm.pdf(x, mu, sigma), '--', label = 'Gaussian Dist.')

  
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 14, markerscale = 1)




#%%
#PHASE 3: Random Walks & Preferential Attachment

#Values for q = 0, 0.5, 0.9

m = 2
added_nodes = 100000

q_set = [0, 0.5, 0.95]
repeat_val = 35






rand_walk_full_vert_count = [[] for i in range(len(q_set))]


full_error_set = [[] for i in range(len(q_set))]
full_error_set_x = [[] for i in range(len(q_set))]


for i in range(len(q_set)):
   

    k_log_bin_err_set = []
    p_vals_bin_err_set = []
   
    for j in range(repeat_val):
        graph_1 = rand_walk_Iterate(added_nodes, m, q_set[i])
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list_phase_3(list(graph_1[5]))
       
        for p in range(len(each_vertex_counter)):
            rand_walk_full_vert_count[i].append(each_vertex_counter[p])
       
        k_log_bin_err, p_vals_bin_err = logbin(each_vertex_counter, 1.2,   True)

        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())
 
   
    max_set_val = 0
    max_i = 0
    for g in range(len(k_log_bin_err_set)):
        if max(k_log_bin_err_set[g]) > max_set_val:
            max_i = g
            max_set_val = max(k_log_bin_err_set[g])
    
    x_vals = np.linspace(0, max_set_val, max_set_val+1).tolist()
    full_error_set_x[i] = np.linspace(0, max_set_val, max_set_val+1)

    errors_set_list = [[x_vals] for p in range(len(k_log_bin_err_set))]
    
    p_vals_set =[[0 for p in range(len(x_vals))] for p in range(len(k_log_bin_err_set))] 
    
    
    for c in range(len(k_log_bin_err_set)):
        for p in range(len(errors_set_list[c][0])):
            for j in range(len(k_log_bin_err_set[c])):
                if errors_set_list[c][0][p] == k_log_bin_err_set[c][j]:
                    p_vals_set[c][p] = p_vals_bin_err_set[c][j]

            
    
    fin_err_set = [0 for i in range(len(x_vals))]
    
    for c in range(len(fin_err_set)):
        test_run = 0
        new_set = []
        for p in range(len(p_vals_set)):
            new_set.append(p_vals_set[p][c])

            
        new_err = np.std(new_set) / len(new_set)
        fin_err_set[c] = new_err
    
    full_error_set[i] = fin_err_set


    print(i, ' done')      


#%%

fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

q_color_set = ['blue', 'orange', 'green', 'red', 'purple']
plt.rcParams.update({'font.size': 12})
plt.xlabel('k', fontsize = 20)
plt.ylabel('p(k, q)', fontsize = 20)


q_labels = ['q = 0', 'q = 0.5', 'q = 0.9']

for i in range(len(rand_walk_full_vert_count)):
    k_prop_log, n_k_log = logbin(rand_walk_full_vert_count[i], 1.2)
    plt.loglog(k_prop_log, n_k_log, '--', ms = 4, label = q_labels[i])
    
    temp_error_set = [0 for i in range(len(k_prop_log))]
    
    for c in range(len(k_prop_log)):
        for j in range(len(full_error_set_x[i])):
            if k_prop_log[c] == full_error_set_x[i][j]:
                temp_error_set[c] =  full_error_set[i][j] 
                    
                    
    plt.errorbar(k_prop_log, n_k_log,  yerr = temp_error_set, xerr = 0 ,ms = 3,capsize = 3,fmt = 'o',  label = q_labels[i])

    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 12, markerscale = 1)
   
    
#%%
#For large range of q from 0 to 0.95
    
m = 2
added_nodes = 100000

q_set = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

repeat_val = 35






rand_walk_full_vert_count = [[] for i in range(len(q_set))]


full_error_set = [[] for i in range(len(q_set))]
full_error_set_x = [[] for i in range(len(q_set))]


for i in range(len(q_set)):
   

    k_log_bin_err_set = []
    p_vals_bin_err_set = []
   
    for j in range(repeat_val):
        graph_1 = rand_walk_Iterate(added_nodes, m, q_set[i])
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list_phase_3(list(graph_1[5]))
       
        for p in range(len(each_vertex_counter)):
            rand_walk_full_vert_count[i].append(each_vertex_counter[p])
       
        k_log_bin_err, p_vals_bin_err = logbin(each_vertex_counter, 1.2,   True)

        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())
 
   
    max_set_val = 0
    max_i = 0
    for g in range(len(k_log_bin_err_set)):
        if max(k_log_bin_err_set[g]) > max_set_val:
            max_i = g
            max_set_val = max(k_log_bin_err_set[g])
    
    x_vals = np.linspace(0, max_set_val, max_set_val+1).tolist()
    full_error_set_x[i] = np.linspace(0, max_set_val, max_set_val+1)

    errors_set_list = [[x_vals] for p in range(len(k_log_bin_err_set))]
    
    p_vals_set =[[0 for p in range(len(x_vals))] for p in range(len(k_log_bin_err_set))] 
    
    
    for c in range(len(k_log_bin_err_set)):
        for p in range(len(errors_set_list[c][0])):
            for j in range(len(k_log_bin_err_set[c])):
                if errors_set_list[c][0][p] == k_log_bin_err_set[c][j]:
                    p_vals_set[c][p] = p_vals_bin_err_set[c][j]

            
    
    fin_err_set = [0 for i in range(len(x_vals))]
    
    for c in range(len(fin_err_set)):
        test_run = 0
        new_set = []
        for p in range(len(p_vals_set)):
            new_set.append(p_vals_set[p][c])

            
        new_err = np.std(new_set) / len(new_set)
        fin_err_set[c] = new_err
    
    full_error_set[i] = fin_err_set


    print(i, ' done')      


#%%

fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

q_color_set = ['blue', 'orange', 'green', 'red', 'purple']
plt.rcParams.update({'font.size': 12})
plt.xlabel('k', fontsize = 20)
plt.ylabel('p(k, q)', fontsize = 20)


q_labels = ['q = 0','q = 0.1', 'q = 0.2','q = 0.3','q = 0.4', 'q = 0.5', 'q = 0.6','q = 0.7','q = 0.8','q = 0.9', 'q = 0.95']

for i in range(len(rand_walk_full_vert_count)):
    k_prop_log, n_k_log = logbin(rand_walk_full_vert_count[i], 1.2)
    plt.loglog(k_prop_log, n_k_log, '--', ms = 4, label = q_labels[i])
    
    temp_error_set = [0 for i in range(len(k_prop_log))]
    
    for c in range(len(k_prop_log)):
        for j in range(len(full_error_set_x[i])):
            if k_prop_log[c] == full_error_set_x[i][j]:
                temp_error_set[c] =  full_error_set[i][j] 
                    
                    
    plt.errorbar(k_prop_log, n_k_log,  yerr = temp_error_set, xerr = 0 ,ms = 3,capsize = 3,fmt = 'o',  label = q_labels[i])

    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 12, markerscale = 1)
   
    
#%%
#for q = 0.5, 0.55, 0.6
    

    
m = 2
added_nodes = 100000

q_set = [0.5, 0.55, 0.6]

repeat_val = 2






rand_walk_full_vert_count = [[] for i in range(len(q_set))]


full_error_set = [[] for i in range(len(q_set))]
full_error_set_x = [[] for i in range(len(q_set))]


for i in range(len(q_set)):
   

    k_log_bin_err_set = []
    p_vals_bin_err_set = []
   
    for j in range(repeat_val):
        graph_1 = rand_walk_Iterate(added_nodes, m, q_set[i])
        k_2, n_k_set_2, instances_of_each_vertex, each_vertex_counter = n_k_list_phase_3(list(graph_1[5]))
       
        for p in range(len(each_vertex_counter)):
            rand_walk_full_vert_count[i].append(each_vertex_counter[p])
       
        k_log_bin_err, p_vals_bin_err = logbin(each_vertex_counter, 1.2,   True)

        k_log_bin_err_set.append(k_log_bin_err.tolist())
        p_vals_bin_err_set.append(p_vals_bin_err.tolist())
 
   
    max_set_val = 0
    max_i = 0
    for g in range(len(k_log_bin_err_set)):
        if max(k_log_bin_err_set[g]) > max_set_val:
            max_i = g
            max_set_val = max(k_log_bin_err_set[g])
    
    x_vals = np.linspace(0, max_set_val, max_set_val+1).tolist()
    full_error_set_x[i] = np.linspace(0, max_set_val, max_set_val+1)

    errors_set_list = [[x_vals] for p in range(len(k_log_bin_err_set))]
    
    p_vals_set =[[0 for p in range(len(x_vals))] for p in range(len(k_log_bin_err_set))] 
    
    
    for c in range(len(k_log_bin_err_set)):
        for p in range(len(errors_set_list[c][0])):
            for j in range(len(k_log_bin_err_set[c])):
                if errors_set_list[c][0][p] == k_log_bin_err_set[c][j]:
                    p_vals_set[c][p] = p_vals_bin_err_set[c][j]

            
    
    fin_err_set = [0 for i in range(len(x_vals))]
    
    for c in range(len(fin_err_set)):
        test_run = 0
        new_set = []
        for p in range(len(p_vals_set)):
            new_set.append(p_vals_set[p][c])

            
        new_err = np.std(new_set) / len(new_set)
        fin_err_set[c] = new_err
    
    full_error_set[i] = fin_err_set


    print(i, ' done')      


#%%

fig = plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 16})

q_color_set = ['blue', 'orange', 'green', 'red', 'purple']
plt.rcParams.update({'font.size': 12})
plt.xlabel('k', fontsize = 20)
plt.ylabel('p(k, q)', fontsize = 20)


q_labels = ['q = 0.5', 'q = 0.55', 'q = 0.6']

for i in range(len(rand_walk_full_vert_count)):
    k_prop_log, n_k_log = logbin(rand_walk_full_vert_count[i], 1.2)
    plt.loglog(k_prop_log, n_k_log, '--', ms = 4, label = q_labels[i])
    
    temp_error_set = [0 for i in range(len(k_prop_log))]
    
    for c in range(len(k_prop_log)):
        for j in range(len(full_error_set_x[i])):
            if k_prop_log[c] == full_error_set_x[i][j]:
                temp_error_set[c] =  full_error_set[i][j] 
                    
                    
    plt.errorbar(k_prop_log, n_k_log,  yerr = temp_error_set, xerr = 0 ,ms = 3,capsize = 3,fmt = 'o',  label = q_labels[i])

    plt.legend(numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 12, markerscale = 1)
   
    

