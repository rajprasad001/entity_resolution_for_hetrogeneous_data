import pandas as pd
import numpy as np
import csv
import collections
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from ast import  literal_eval
from datetime import datetime
slim = tf.contrib.slim

row_size=300# size of the dimension we are limiting our embedding
code_length=64 #6#16# can be modified depending upon computation and coverage
external_dropout=0.8 #Percentage of things that will remain
internal_dropout=0.65 #Percentage of things that will remain
optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001)
batch_size= 64
total_it=200000 #Total number of iterations, each iteration has a number of batches and gives a report at the end
num_epochs=2 #Epochs per batch
report_interval=1000
similarity_weight=1.5 #This is a hyper-parameter we can use to boost the importance of the similarity in the computation
use_probs=False
df = pd.read_csv("C:/Users/anujp/Desktop/sort/Entity_Resolution_Project/data/csv_files/2020-03-27  12-46cleaned_all_ds_spec_id_page_title_embedded.csv", header=0)
#df2 = pd.read_csv(location+"ProductIdsPageTitles.csv", header=0)
#df2.drop(df2.columns.difference(['spec_id']), 1, inplace=True)
#df1.drop(columns=[100], inplace=True)
#df=  pd.concat([df2.reset_index(drop=True), df1.reset_index(drop=True)], axis= 1)
# Our first dataset is now stored in a Pandas Dataframe

#Here we check the schema, and its length
print("Schema: "+str(df.columns))
print("Number of rows: "+str(len(df)))
print(df.head())
match_pair_orig=pd.read_csv("df_transative_pairs.csv", header=0)
match_pair_orig=match_pair_orig.applymap(str).groupby('Object')['Pairs'].apply(set).to_dict()
match_pair=dict()
for k in match_pair_orig:
  match_pair[k.replace("'","")]=set([str(item).replace("'","") for item in tuple(match_pair_orig[k])[0].replace("[","").replace("]","").replace(" ","").split(",")])
print(tuple(match_pair.keys())[0])
non_match_pair_orig=pd.read_csv("df_transative_nonmatch.csv", header=0)
non_match_pair_orig=non_match_pair_orig.applymap(str).groupby('Object')['Non_Pairs'].apply(set).to_dict()
non_match_pair=dict()
for k in non_match_pair_orig:
  non_match_pair[k.replace("'","")]=set([str(item).replace("'","") for item in tuple(non_match_pair_orig[k])[0].replace("[","").replace("]","").replace(" ","").split(",")])
print(tuple(non_match_pair.keys())[0])

setA = set(match_pair)
setB= set(non_match_pair)
setC=[]

setC=setA.intersection(setB)
print(len(setC))

emb_dict={}
for index, row in df.iterrows():
  emb_dict[row["spec_id"].replace("'","")]=[float(row[str(item)]) for item in range(0,100)]
print(tuple(emb_dict.keys())[0])

split_boundary=33 #Write 33 for pickign 33% of the data for testing and the remaining for training.
test_set_matches=set()
test_set_non_matches=set()
for first_item in match_pair.keys():
  for second_item in match_pair[first_item]:
    if first_item < second_item:#Our tuple formation criteria
      if random.randrange(1,101) <=split_boundary:
        test_set_matches.add((first_item,second_item))

for first_item in non_match_pair.keys():
  for second_item in non_match_pair[first_item]:
    if first_item < second_item:#Our tuple formation criteria
      if random.randrange(1,101) <=split_boundary:
        test_set_non_matches.add((first_item,second_item))

def deep_hash_network(code_length, network_type, input):
    net = tf.nn.dropout(slim.fully_connected(input, 64, activation_fn=tf.nn.relu), external_dropout)
    net = tf.nn.dropout(slim.fully_connected(net, 32, activation_fn=tf.nn.relu), internal_dropout)#, weights_regularizer=slim.l2_regularizer(1e-8))  
    #net = slim.fully_connected(net, 32, activation_fn=tf.nn.relu)#, weights_regularizer=slim.l2_regularizer(1e-8))  
    hash_code = tf.nn.dropout(slim.fully_connected(net, code_length, activation_fn=None), 1.0)#  <- why?
    return network_type(hash_code)

our_net=deep_hash_network
print("Network function defined")

tf_device='/gpu:*'
print("Device defined")
    
shape=(1,row_size) 
print("I/O shapes defined")
    
def _network_template(state):# <-Purpose?
    return our_net(code_length, collections.namedtuple('DQH_network', ['hash_values']), state)
print("Network wrapper function defined")
    
batch_outputs1=[]# match
batch_outputs2=[]# match
batch_outputs3=[]# non_match

def _build_network():
    global batch_outputs1, batch_outputs2, batch_outputs3
    net= tf.make_template('network', _network_template)
    batch_outputs1=tf.clip_by_value(net(states1_ph),-1.,1.)#<-setting min and max value?
    batch_outputs2=tf.clip_by_value(net(states2_ph),-1.,1.)
    batch_outputs3=tf.clip_by_value(net(states3_ph),-1.,1.)      
print("Network forward pass function defined")
    

#RMSPRop
def _build_train_op(): 
  #This defines our training operation, based on: Li, Wu-Jun, Sheng Wang, and Wang-Cheng Kang. 
  #"Feature learning based deep supervised hashing with pairwise labels." 
  #arXiv preprint arXiv:1511.03855 (2015). However we extend it to a triple, because it worked better for us.
  theta=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs1[0],batch_outputs2[0]),1),2)
  theta2=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs1[0],batch_outputs3[0]),1),2)
  theta5=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs2[0],batch_outputs3[0]),1),2)
  
  sim_loss=-(
      -tf.math.log(1+tf.math.exp(theta))+similarity_weight*theta 
      
  )

  disim_loss2=-(
      -tf.math.log(1+tf.math.exp(theta2))
      
  )

  disim_loss5=-(
      -tf.math.log(1+tf.math.exp(theta5))
      
  )
  loss=sim_loss+disim_loss2+disim_loss5
  print_op=tf.print(tf.reduce_mean(loss, axis=0))
  with tf.control_dependencies([]):#Write print_op in there to print loss
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs] #We clip by value to avoid exploiding gradients -10 to 10
    return optimizer.apply_gradients(capped_gvs)
              
print("Network backprop pass function (loss definition and minimization criteria) defined")
    
size_of_embedding=row_size
with tf.device(tf_device):
    batch_outputs1=tf.placeholder(tf.float32, name='bo1_ph')
    batch_outputs2=tf.placeholder(tf.float32, name='bo2_ph')
    batch_outputs3=tf.placeholder(tf.float32, name='bo3_ph')
    states1_ph = tf.placeholder(tf.float32, (None,size_of_embedding), name='state1_ph')  
    states2_ph = tf.placeholder(tf.float32, (None,size_of_embedding), name='state2_ph')
    states3_ph = tf.placeholder(tf.float32, (None,size_of_embedding), name='state3_ph')
    net= _build_network()
    _train_op = _build_train_op()    
print("Device selected and variables initialized")
    
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_sess = tf.Session('', config=config)
init_op = tf.initialize_all_variables()
_sess.run(init_op)
print("Tensorflow session initalized")

choices_selected=dict()#UPDATE LOGIC
for item in list(match_pair.keys()):
    choices_selected[item]=1
sumcs=0
for item in choices_selected.keys():#PUT THIS ALSO DOWN
    sumcs+=choices_selected[item]
probs=[choices_selected[key]/sumcs for key in list(match_pair.keys())]
for num in range(0,total_it):
    #if num%100==0:
    #    print(num)
    if use_probs:
        KeysSelected = random.choices(list(match_pair.keys()), k=batch_size, weights=probs)
        weights=[]
        for a in KeysSelected:
            mplist=[choices_selected[c] for c in list(match_pair[a])]
            sum_mplist=sum(mplist)
            mplist=[c/sum_mplist for c in mplist]
            weights.append(mplist)
        MatchesSelected = [random.choices(list(match_pair[KeysSelected[a]]), k=1, weights=weights[a])[0].replace("'","") for a in range(0,len(KeysSelected))]
    else:
        KeysSelected = random.choices(list(match_pair.keys()), k=batch_size)
        MatchesSelected = [random.choice(list(match_pair[KeysSelected[a]])).replace("'","") for a in range(0,len(KeysSelected))]
    firsts=[]
    seconds=[]
    non_matches=[]
    for i in range(0,batch_size):
        small_key=None
        big_key=None
        if KeysSelected[i]<=MatchesSelected[i]:
          small_key=KeysSelected[i]
          big_key=MatchesSelected[i]
        else:
          big_key=KeysSelected[i]
          small_key=MatchesSelected[i]
        if (KeysSelected[i] in non_match_pair) and (KeysSelected[i] in emb_dict) and (MatchesSelected[i] in emb_dict) and not (small_key,big_key) in test_set_matches:
            we_selected= random.choice(tuple(non_match_pair[KeysSelected[i]])).replace("'","")
            #if not we_selected in emb_dict:
            #    we_selected="'"+we_selected+"'"
            if small_key<=we_selected:
              big_key=we_selected
            else:
              big_key=small_key
              small_key=we_selected
            if we_selected in emb_dict and not (small_key, big_key) in test_set_non_matches:    
              non_matches.append(emb_dict[we_selected])
              firsts.append(emb_dict[KeysSelected[i]])
              seconds.append(emb_dict[MatchesSelected[i]]) 
    #print(len(firsts))
    if (len(firsts)==0):
      print("No item in match and embeddings")       
    for epoch in range(0,num_epochs):
        [result]=_sess.run([_train_op], feed_dict={states1_ph: np.array(firsts,dtype=np.float64), states2_ph: np.array(seconds,dtype=np.float64), states3_ph: np.array(non_matches,dtype=np.float64)})
    if num%report_interval==0:
      test_set=[]
      keys=list(emb_dict.keys())
      for item in keys:
         test_set.append(emb_dict[item])
      b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(np.array(test_set,dtype=np.float64),dtype=np.float64)})[0])
      hash_code_dict={}
      key_2_hash_code_dict={}
      for item in range(0,len(keys)):
        hash_code="".join([str(int(a)) for a in b1[item]])
        hash_code=hash_code.replace("-1","0")
        if not hash_code in hash_code_dict:
          hash_code_dict[hash_code]=set()
        set_to_use=hash_code_dict[hash_code]
        set_to_use.add(keys[item].replace("'",""))
        hash_code_dict[hash_code]=set_to_use
        key_2_hash_code_dict[keys[item].replace("'","")]=hash_code
      #print(len(key_2_hash_code_dict.keys()))
      print("Iteration: "+str(num))
      print("Number of hash codes: "+str(len(hash_code_dict.keys())))
      total_matches=0
      good_matches=0
      faulty_items=set()
      now=True
      if use_probs:
        choices_selected=dict()#UPDATE LOGIC
        for item in list(match_pair.keys()):
          choices_selected[item]=1
      for k in match_pair.keys():
        if not k in key_2_hash_code_dict:
          if not k in faulty_items:
            faulty_items.add(k)
            print("Adding "+k+", from key_2_hash_code")
        else:
          kscode= key_2_hash_code_dict[k]
          for item in list(match_pair[k]):            
            if not item in key_2_hash_code_dict:
              if not item in faulty_items:
                faulty_items.add(item)
            else:
              total_matches+=1
              if kscode == key_2_hash_code_dict[item]:
                good_matches+=1
              elif use_probs:
                choices_selected[item]+=1
                choices_selected[k]+=1
      if use_probs:
        sumcs=0
        for item in choices_selected.keys():#PUT THIS ALSO DOWN
          sumcs+=choices_selected[item]
        probs=[choices_selected[key]/sumcs for key in list(match_pair.keys())]
      coverage=100*good_matches/total_matches
      print("Coverage is: "+str(coverage))
      print("Non products keys: "+str(len(faulty_items)))
      computation=0
      for key in hash_code_dict:
        if len(hash_code_dict[key])>1:
          computation+=(len(hash_code_dict[key])*len(hash_code_dict[key]))
      computation=100*computation/(len(key_2_hash_code_dict.keys())*len(key_2_hash_code_dict.keys()))
      print("Computation is "+str(computation))


from datetime import datetime
dateTimeObj = datetime.now().replace(second=0, microsecond=0)
f = open(str(dateTimeObj)+"_Codes_"+str(len(hash_code_dict.keys()))+"_Cov_"+str(coverage)+"_Comp_"+"%.2f" % computation+"_generated_hashes_codelength64_simweight1_5_no_probs.csv",'w')
for item in key_2_hash_code_dict.keys():
  f.write(item+","+key_2_hash_code_dict[item]+"\n")
f.close()

counter=0
for item in sorted(list(faulty_items)):
  print(str(counter)+"-"+item)
  counter+=1
