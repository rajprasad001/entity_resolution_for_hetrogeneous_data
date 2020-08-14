import pandas as pd
import numpy as np
import collections
import random
import tensorflow as tf
import csv
slim = tf.contrib.slim
    
for row_size in [901]:#, 901]:#,901
  for code_length in [32]:#,48,64,128]:#,48,64,128
    batch_size= 32
    
    use_subset=False
    k_for_gt=858
    k_for_nongt=2129
    
    total_it=1000 #Total number of iterations, each iteration has a number of batches and gives a report at the end
    num_batches=50 #Number of batches chosen between report creation
    num_epochs=2 #Epochs per batch
    
    df = pd.read_csv("GoogleProductsEmbedded.csv", header=None)
    
    # Our first dataset is now stored in a Pandas Dataframe
    
    #Here we check the schema, and its length
    print("Schema: "+str(df.columns))
    print("Number of rows: "+str(len(df)))
    
    df2 = pd.read_csv("AmazonEmbedded.csv", header=None)
    
    # Our second dataset is now stored in a Pandas Dataframe
    
    #Here we check the schema, and its length
    print("Schema: "+str(df2.columns))
    print("Number of rows: "+str(len(df2)))
    
    df3 = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv")
    
    # Our ground truth dataset is now stored in a Pandas Dataframe
    
    #Here we check the schema, and its length
    print("Schema: "+str(df3.columns))
    print("Number of rows: "+str(len(df3)))
    
    counter=0
    GoogleDict=dict()
    for item in df[0]:
      GoogleDict[item]=counter
      counter+=1
    
    counter=0
    AmazonDict=dict()
    for item in df2[0]:
      AmazonDict[item]=counter
      counter+=1
    
    GTDict=dict()
    for index, row in df3.iterrows():
      if (row['idAmazon'] in GTDict):
        arr=GTDict[row['idAmazon']]
        arr.add(row['idGoogleBase'])
        GTDict[row['idAmazon']]=arr
      else:
        newset=set()
        newset.add(row['idGoogleBase'])
        GTDict[row['idAmazon']]=newset
    
    if use_subset:
      akeys= random.choices(list(GTDict.keys()), k=k_for_gt)
      gkeys=[]
      for k in akeys:
        gkeys.append(random.choice(list(GTDict[k])))
      for k in range(0,k_for_nongt):
        gkeys.append(df.iloc[random.choice(range(0,len(df))),0])
        akeys.append(df2.iloc[random.choice(range(0,len(df2))),0])
    
      print(gkeys)
      print(akeys)
      df=df[df[0].isin(gkeys)]
      df2=df2[df2[0].isin(akeys)]
    
      counter=0
      GoogleDict=dict()
      for item in df[0]:
        GoogleDict[item]=counter
        counter+=1
      
      print(GoogleDict)
      print("Keys: ",len(GoogleDict))
      counter=0
      AmazonDict=dict()
      for item in df2[0]:
        AmazonDict[item]=counter
        counter+=1
      
      print(AmazonDict)
      print("Keys: ",len(AmazonDict))

      GTDict=dict()
      for index, row in df3.iterrows():
        if (row['idAmazon'] in GTDict and row['idAmazon'] in akeys and row['idGoogleBase'] in gkeys):
          arr=GTDict[row['idAmazon']]
          arr.add(row['idGoogleBase'])
          GTDict[row['idAmazon']]=arr
        elif row['idAmazon'] in akeys and row['idGoogleBase'] in gkeys:
          newset=set()
          newset.add(row['idGoogleBase'])
          GTDict[row['idAmazon']]=newset
      print(GTDict)    
      print("Keys: ",len(GTDict))

    def deep_hash_network(code_length, network_type, input):
      net = slim.fully_connected(input, 5120, activation_fn=tf.nn.relu)
      #net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(1e-8))  
      hash_code = slim.fully_connected(net, code_length, activation_fn=None)
      return network_type(hash_code)
    
    
    our_net=deep_hash_network
    print("Network function defined")
    
    tf_device='/gpu:*'
    print("Device defined")
    
    shape=(1,row_size) 
    print("I/O shapes defined")
    
    def _network_template(state):
      return our_net(code_length, collections.namedtuple('DQH_network', ['hash_values']), state)
    
    print("Network wrapper function defined")
    
    batch_outputs1=[]
    batch_outputs2=[]
    batch_outputs3=[]
    def _build_network():
      global batch_outputs1, batch_outputs2, batch_outputs3
      net= tf.make_template('network', _network_template)
      batch_outputs1=tf.clip_by_value(net(states1_ph),-1.,1.)
      batch_outputs2=tf.clip_by_value(net(states2_ph),-1.,1.)
      batch_outputs3=tf.clip_by_value(net(states3_ph),-1.,1.)
      
    print("Network forward pass function defined")
    
    #optimizer=tf.contrib.opt.AdaMaxOptimizer(learning_rate=0.0001)
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001)

    def _build_train_op():
      theta=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs1[0],batch_outputs2[0]),1),2)
      theta2=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs1[0],batch_outputs3[0]),1),2)
      theta5=tf.divide(tf.reduce_sum(tf.multiply(batch_outputs2[0],batch_outputs3[0]),1),2)
      
      sim_loss=-tf.reduce_sum(
         -tf.math.log(1+tf.math.exp(theta))+3*theta
         , 0
      )
    
      disim_loss2=-tf.reduce_sum(
          -tf.math.log(1+tf.math.exp(theta2))
          , 0
      )
    
      disim_loss5=-tf.reduce_sum(
          -tf.math.log(1+tf.math.exp(theta5))
          , 0
      )
      loss=sim_loss+disim_loss2+disim_loss5
      with tf.control_dependencies([]):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
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
    
    
    AmazDict=dict()
    GoogDict=dict()
    GoogHash=dict()
    GoogKey2Hash=dict()
    AmazKey2Hash=dict()
    AmazHash=dict()
    amzn=[]
    googl=[]
    coverage=0
    summary=dict()
    for num in range(0,total_it):
      ANeedingCover=[]
      cover_count=0
      comp_count=0
      if len(AmazKey2Hash) == 0:
        ANeedingCover=list(GTDict.keys())
      else:
        for a in list(GTDict.keys()):
          completely_covered=True
          for g in GTDict[a]:
            if GoogKey2Hash[g]!=AmazKey2Hash[a]: #Here we check for cover
              completely_covered=False
              break
          if not completely_covered:
            ANeedingCover.append(a)
            cover_count+=1
          elif sum([1 for x in GoogKey2Hash if GoogKey2Hash[x]==AmazKey2Hash[a]])>(len(GTDict[a])+0.005*len(df2)): #Here we check to reduce computation
            ANeedingCover.append(a)
            comp_count+=1
      if len(ANeedingCover)==0:
        ANeedingCover=list(GTDict.keys())
      print("# of ANeedingCover: ",len(ANeedingCover))
      print("# of Actually needing cover: ",cover_count)
      print("# of Actually needing to reduce comp: ",comp_count)
      for rep in range (0,num_batches): 
        google=[]
        amazon=[]
        disim=[]
        GHash=dict()
        distant_count=0
        but_similar=0
        AmazonSelected = random.choices(range(0,len(df2)), k=batch_size)
        GoogleSelected = random.choices(range(0,len(df)), k=batch_size)
        for i in range(0,batch_size):
          if random.choice([True,False]):
            flip=True
          else:
            flip=False
          if random.choice([True,False]):
            if len(ANeedingCover)<batch_size:
              selection=random.choice(range(0,batch_size))
              if selection<len(ANeedingCover):
                AKey=random.choice(ANeedingCover)
              else:
                AKey=random.choice(list(GTDict.keys()))                
            else:
              AKey=random.choice(ANeedingCover)
          else:
            AKey=random.choice(list(GTDict.keys()))
          if row_size!=901:
            if not flip:
              amazon.append(df2.iloc[AmazonDict[AKey],1:row_size+1])
            else:
              google.append(df2.iloc[AmazonDict[AKey],1:row_size+1])
          else:
            if not flip:
              amazon.append(df2.iloc[AmazonDict[AKey],1:])
            else:
              google.append(df2.iloc[AmazonDict[AKey],1:])
          GKey=random.choice(list(GTDict[AKey]))
          if row_size!=901:
            if not flip:
              google.append(df.iloc[GoogleDict[GKey],1:row_size+1])
            else:
              amazon.append(df.iloc[GoogleDict[GKey],1:row_size+1])
          else:
            if not flip:
              google.append(df.iloc[GoogleDict[GKey],1:])
            else:
              amazon.append(df.iloc[GoogleDict[GKey],1:])
          if row_size!=901:
            if random.choice([True,False]):
              passed=False
              if GKey in GoogKey2Hash:
                choice= random.choice(list(GoogHash[GoogKey2Hash[GKey]]))
                if choice!=GKey and not choice in list(GTDict[AKey]):
                  disim.append(df.iloc[GoogleDict[choice],1:row_size+1])
                  passed=True
              if not passed:
                choice = random.choice(range(0,len(df)))
                while (df.iloc[choice,0] in list(GTDict[AKey])):
                  choice = random.choice(range(0,len(df)))
                disim.append(df.iloc[choice,1:row_size+1])
                distant_count+=1
            else:
              passed=False
              if AKey in AmazKey2Hash:
                choice= random.choice(list(AmazHash[AmazKey2Hash[AKey]]))
                if choice!=AKey and ((choice not in GTDict) or (GKey not in GTDict[choice])):
                  disim.append(df2.iloc[AmazonDict[choice],1:row_size+1])
                  passed=True
              if not passed:
                choice = random.choice(range(0,len(df2)))
                while (df2.iloc[choice,0]==AKey or ((df2.iloc[choice,0] in GTDict) and (GKey in GTDict[df2.iloc[choice,0]]))):
                  choice = random.choice(range(0,len(df2)))
                disim.append(df2.iloc[choice,1:row_size+1])
                distant_count+=1
          else:
            if random.choice([True,False]):
              passed=False
              if GKey in GoogKey2Hash:
                choice= random.choice(list(GoogHash[GoogKey2Hash[GKey]]))
                if choice!=GKey and not choice in list(GTDict[AKey]):
                  disim.append(df.iloc[GoogleDict[choice],1:])
                  passed=True
              if not passed:
                choice = random.choice(range(0,len(df)))
                while (df.iloc[choice,0] in list(GTDict[AKey])):
                  choice = random.choice(range(0,len(df)))
                disim.append(df.iloc[choice,1:])
                distant_count+=1
            else:
              passed=False
              if AKey in AmazKey2Hash:
                choice= random.choice(list(AmazHash[AmazKey2Hash[AKey]]))
                if choice!=AKey and (choice not in GTDict or GKey not in GTDict[choice]):
                  disim.append(df2.iloc[AmazonDict[choice],1:])
                  passed=True
              if not passed:
                choice = random.choice(range(0,len(df2)))
                while (df2.iloc[choice,0]==AKey or ((df2.iloc[choice,0] in GTDict) and (GKey in GTDict[df2.iloc[choice,0]]))):
                  choice = random.choice(range(0,len(df2)))
                disim.append(df2.iloc[choice,1:])
                distant_count+=1
        if rep%10==0:
          print("Reporting Episode: ",num,"- It: ",rep, "- Batch size: ",len(amazon),"- Distant Count: ",distant_count)
        for epoch in range(0,num_epochs):
          [result]=_sess.run([_train_op], feed_dict={states1_ph: np.array(amazon,dtype=np.float64), states2_ph: np.array(google,dtype=np.float64), states3_ph: np.array(disim,dtype=np.float64)})
    
    
      real_matches=0
      hash_matches=0
      close_matches=0
      close_matches2=0
      close_matches3=0
      close_matches4=0
      close_matches5=0
      close_matches6=0
      close_matches7=0
      close_matches8=0
      close_matches9=0
      close_matches10=0
      close_matches11=0
      amzn=[]
      googl=[]
      aids=[]
      gids=[]
      GoogHash=dict()
      GoogKey2Hash=dict()
      AmazKey2Hash=dict()
      AmazHash=dict()
      for amazon_item in list(GTDict.keys()):
        for match in list(GTDict[amazon_item]):
          real_matches+=1
          if row_size!=901:
            amzn.append(df2.iloc[AmazonDict[amazon_item],1:row_size+1])
            googl.append(df.iloc[GoogleDict[match],1:row_size+1])
          else:
            amzn.append(df2.iloc[AmazonDict[amazon_item],1:])
            googl.append(df.iloc[GoogleDict[match],1:])
          if (real_matches%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph:np.array(googl, dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            b2= np.sign(_sess.run(batch_outputs2, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              if(np.array_equal(b1[row],b2[row])):
                hash_matches+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=2):
                close_matches+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=3):
                close_matches2+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=4):
                close_matches3+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=5):
                close_matches4+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=6):
                close_matches5+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=7):
                close_matches6+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=8):
                close_matches7+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=9):
                close_matches8+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=10):
                close_matches9+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=20):
                close_matches10+=1
              elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=32):
                close_matches11+=1            
            googl.clear()
            amzn.clear()
      if len(amzn)>0:
        b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
        b2= np.sign(_sess.run(batch_outputs2, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
        for row in range(0,len(b1)):
          if(np.array_equal(b1[row],b2[row])):
            hash_matches+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=2):
            close_matches+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=3):
            close_matches2+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=4):
            close_matches3+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=5):
            close_matches4+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=6):
            close_matches5+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=7):
            close_matches6+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=8):
            close_matches7+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=9):
            close_matches8+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=10):
            close_matches9+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=20):
            close_matches10+=1
          elif(sum([1 for i, j in zip(b1[row], b2[row]) if i != j])<=32):
            close_matches11+=1
        googl.clear()
        amzn.clear()
      print("Coverage (%) with hash matches (0):",100*(hash_matches)/real_matches)
      print("Coverage (%) with close matches (2+):",100*(hash_matches+close_matches)/real_matches)
      print("Coverage (%) with neighboring matches (3+):",100*(hash_matches+close_matches+close_matches2)/real_matches)
      print("Coverage (%) with neighboring matches (4+):",100*(hash_matches+close_matches+close_matches2+close_matches3)/real_matches)
      print("Coverage (%) with neighboring matches (5+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4)/real_matches)
      print("Coverage (%) with neighboring matches (6+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5)/real_matches)
      print("Coverage (%) with neighboring matches(7+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6)/real_matches)
      print("Coverage (%) with neighboring matches(8+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6+close_matches7)/real_matches)
      print("Coverage (%) with neighboring matches(9+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6+close_matches7+close_matches8)/real_matches)
      print("Coverage (%) with neighboring matches(10+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6+close_matches7+close_matches8+close_matches9)/real_matches)
      print("Coverage (%) with neighboring matches(20+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6+close_matches7+close_matches8+close_matches9+close_matches10)/real_matches) 
      print("Coverage (%) with neighboring matches(32+):",100*(hash_matches+close_matches+close_matches2+close_matches3+close_matches4+close_matches5+close_matches6+close_matches7+close_matches8+close_matches9+close_matches10+close_matches11)/real_matches)
      print("Coverage (%) with close matches (2+):",100*(close_matches)/real_matches)
      print("Coverage (%) with neighboring matches (3+):",100*(close_matches2)/real_matches)
      print("Coverage (%) with neighboring matches (4+):",100*(close_matches3)/real_matches)
      print("Coverage (%) with neighboring matches (5+):",100*(close_matches4)/real_matches)
      print("Coverage (%) with neighboring matches (6+):",100*(close_matches5)/real_matches)
      print("Coverage (%) with neighboring matches(7+):",100*(close_matches6)/real_matches)
      print("Coverage (%) with neighboring matches(8+):",100*(close_matches7)/real_matches)
      print("Coverage (%) with neighboring matches(9+):",100*(close_matches8)/real_matches)
      print("Coverage (%) with neighboring matches(10+):",100*(close_matches9)/real_matches)
      print("Coverage (%) with neighboring matches(20+):",100*(close_matches10)/real_matches) 
      print("Coverage (%) with neighboring matches(32+):",100*(close_matches11)/real_matches)
      summary["Coverage"]=100*hash_matches/real_matches
    
      if True:
        cartesian_product=len(AmazonDict.keys())*len(GoogleDict.keys())
        AmazDict=dict()
        GoogDict=dict()
        amzn=[]
        googl=[]
    
    
        for i in range(0,len(df2)):
          aids.append(df2.iloc[i,0])
          if row_size!=901:
            amzn.append(df2.iloc[i,1:row_size+1])
          else:
            amzn.append(df2.iloc[i,1:])
          if(len(amzn)%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(amzn,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              key=np.array2string(b1[row])
              key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
              if not key in AmazDict:
                AmazDict[key]=1
              else:
                AmazDict[key]+=1
              if not key in AmazHash:
                AmazHash[key]=set()
              AmazHash[key].add(aids[row])
              AmazKey2Hash[aids[row]]=key
            aids.clear()
            amzn.clear()
    
        if(len(amzn)>0):
          b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(amzn,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
          for row in range(0,len(b1)):
            key=np.array2string(b1[row])
            key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
            if not key in AmazDict:
              AmazDict[key]=1
            else:
              AmazDict[key]+=1
            if not key in AmazHash:
              AmazHash[key]=set()
            AmazHash[key].add(aids[row])
            AmazKey2Hash[aids[row]]=key
        amzn.clear()
        aids.clear()
    
        for i in range(0,len(df)):
          gids.append(df.iloc[i,0])
          if row_size!=901:
            googl.append(df.iloc[i,1:row_size+1])
          else:
            googl.append(df.iloc[i,1:])
          if(len(googl)%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(googl,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(googl,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              key=np.array2string(b1[row])
              key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
              if not key in GoogDict:
                GoogDict[key]=1
              else:
                GoogDict[key]+=1
              if not key in GoogHash:
                GoogHash[key]=set()
              GoogHash[key].add(gids[row])
              GoogKey2Hash[gids[row]]=key
            googl.clear()
            gids.clear()
        
        if (len(googl)>0):
          b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(googl,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(googl,dtype=np.float64)})[0])
          for row in range(0,len(b1)):
            key=np.array2string(b1[row])
            key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
            if not key in GoogDict:
              GoogDict[key]=1
            else:
              GoogDict[key]+=1
            if not key in GoogHash:
              GoogHash[key]=set()
            GoogHash[key].add(gids[row])
            GoogKey2Hash[gids[row]]=key
        googl.clear()
        gids.clear()
    
        hash_product=0
        for key in list(AmazDict.keys()):
          if key in GoogDict:
            hash_product+=(GoogDict[key]*AmazDict[key])
        print("Computation (%):",100*hash_product/cartesian_product)
        summary["Computation"]=100*hash_product/cartesian_product
        print("Key diversity (Google):",len(GoogDict))
        print("Key diversity (Amazon):",len(AmazDict))
        summary["Key diversity (Google)"]=len(GoogDict)
        summary["Key diversity (Amazon)"]=len(AmazDict)
    
    summary["Number of iterations done"]=total_it
    
    with open('exec_summary_features_'+str(row_size)+'_code_length_'+str(code_length)+'_selected_data.csv', 'w') as f:
      for key in sorted(summary.keys()):
        f.write("%s,%s\n"%(key,summary[key]))
    
    with open('google_hash_features_'+str(row_size)+'_code_length_'+str(code_length)+'_selected_data.csv', 'w') as f:
      for key in sorted(GoogKey2Hash.keys()):
        f.write("%s,%s\n"%(key,GoogKey2Hash[key]))
    
    with open('amazon_hash_features_'+str(row_size)+'_code_length_'+str(code_length)+'._selected_datacsv', 'w') as f:
      for key in sorted(AmazKey2Hash.keys()):
        f.write("%s,%s\n"%(key,AmazKey2Hash[key]))
    
    AmazHash=dict()
    GoogHash=dict()
    GoogHash2KeyCount=dict()
    AmazHash2KeyCount=dict()
    counter=0
    for k in list(GTDict.keys()):
      foundSomething=False
      foundKey=0
      for item in list(GTDict[k]):
        if item in GoogHash:
          key=GoogHash[item]
          if not foundSomething:
            foundKey=key
          elif foundKey!=key:
            print("Double find and different keys!")
            exit()
          foundSomething=True
          break
      if not foundSomething:
        key=counter
        counter+=1
      AmazHash[k]=key
      if key in AmazHash2KeyCount:
        AmazHash2KeyCount[key]+=1
      else:
        AmazHash2KeyCount[key]=1
      for item in list(GTDict[k]):
        GoogHash[item]=key
      if key in GoogHash2KeyCount:
        GoogHash2KeyCount[key]+=len(list(GTDict[k]))
      else:
        GoogHash2KeyCount[key]=len(list(GTDict[k]))
    
    for item in range(0,len(df2)):
      if not df2.iloc[item,0] in AmazHash:
        AmazHash[df2.iloc[item,0]]=counter
        counter+=1
    
    for item in range(0,len(df)):
      if not df.iloc[item,0] in GoogHash:
        GoogHash[df.iloc[item,0]]=counter
        counter+=1
    
    print("Brute force Number of keys used: ",counter) 
    
    real_matches=0
    hash_matches=0
    for amazon_item in list(GTDict.keys()):
      for match in list(GTDict[amazon_item]):
        real_matches+=1
        if(GoogHash[match]==AmazHash[amazon_item]):
          hash_matches+=1
    print("Brute force Coverage (on dataset that we trained on) (%):",100*hash_matches/real_matches)
    
    cartesian_product=len(AmazonDict.keys())*len(GoogleDict.keys())
    hash_product=0
    for key in AmazHash2KeyCount:
      if key in GoogHash2KeyCount:
        hash_product+=GoogHash2KeyCount[key]*AmazHash2KeyCount[key]
    print("Brute force Computation (%):",100*hash_product/cartesian_product)
    
    
    
    if use_subset:
      df = pd.read_csv("GoogleProductsEmbedded.csv", header=None)
      
      # Our first dataset is now stored in a Pandas Dataframe
      
      #Here we check the schema, and its length
      print("Schema: "+str(df.columns))
      print("Number of rows: "+str(len(df)))
      
      df2 = pd.read_csv("AmazonEmbedded.csv", header=None)
      
      # Our second dataset is now stored in a Pandas Dataframe
      
      #Here we check the schema, and its length
      print("Schema: "+str(df2.columns))
      print("Number of rows: "+str(len(df2)))
      
      df3 = pd.read_csv("Amzon_GoogleProducts_perfectMapping.csv")
      
      # Our ground truth dataset is now stored in a Pandas Dataframe
      
      #Here we check the schema, and its length
      print("Schema: "+str(df3.columns))
      print("Number of rows: "+str(len(df3)))
      
      counter=0
      GoogleDict=dict()
      for item in df[0]:
        GoogleDict[item]=counter
        counter+=1
      
      counter=0
      AmazonDict=dict()
      for item in df2[0]:
        AmazonDict[item]=counter
        counter+=1
      
      GTDict=dict()
      for index, row in df3.iterrows():
        if (row['idAmazon'] in GTDict):
          arr=GTDict[row['idAmazon']]
          arr.add(row['idGoogleBase'])
          GTDict[row['idAmazon']]=arr
        else:
          newset=set()
          newset.add(row['idGoogleBase'])
          GTDict[row['idAmazon']]=newset
      
      
      real_matches=0
      hash_matches=0
      amzn=[]
      googl=[]
      aids=[]
      gids=[]
      GoogHash=dict()
      GoogKey2Hash=dict()
      AmazKey2Hash=dict()
      AmazHash=dict()
      for amazon_item in list(GTDict.keys()):
        for match in list(GTDict[amazon_item]):
          real_matches+=1
          if row_size!=901:
            amzn.append(df2.iloc[AmazonDict[amazon_item],1:row_size+1])
            googl.append(df.iloc[GoogleDict[match],1:row_size+1])
          else:
            amzn.append(df2.iloc[AmazonDict[amazon_item],1:])
            googl.append(df.iloc[GoogleDict[match],1:])
          if (real_matches%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph:np.array(googl, dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            b2= np.sign(_sess.run(batch_outputs2, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              if(np.array_equal(b1[row],b2[row])):
                hash_matches+=1
            googl.clear()
            amzn.clear()
      if len(amzn)>0:
        b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
        b2= np.sign(_sess.run(batch_outputs2, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
        for row in range(0,len(b1)):
          if(np.array_equal(b1[row],b2[row])):
            hash_matches+=1
        googl.clear()
        amzn.clear()
      print("Coverage (%):",100*hash_matches/real_matches)
      summary["Coverage"]=100*hash_matches/real_matches
      
      if True:
        cartesian_product=len(AmazonDict.keys())*len(GoogleDict.keys())
        AmazDict=dict()
        GoogDict=dict()
        amzn=[]
        googl=[]
      
      
        for i in range(0,len(df2)):
          aids.append(df2.iloc[i,0])
          if row_size!=901:
            amzn.append(df2.iloc[i,1:row_size+1])
          else:
            amzn.append(df2.iloc[i,1:])
          if(len(amzn)%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(amzn,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              key=np.array2string(b1[row])
              key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
              if not key in AmazDict:
                AmazDict[key]=1
              else:
                AmazDict[key]+=1
              if not key in AmazHash:
                AmazHash[key]=set()
              AmazHash[key].add(aids[row])
              AmazKey2Hash[aids[row]]=key
            aids.clear()
            amzn.clear()
      
        if(len(amzn)>0):
          b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(amzn,dtype=np.float64), states2_ph: np.array(amzn,dtype=np.float64), states3_ph: np.array(amzn,dtype=np.float64)})[0])
          for row in range(0,len(b1)):
            key=np.array2string(b1[row])
            key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
            if not key in AmazDict:
              AmazDict[key]=1
            else:
              AmazDict[key]+=1
            if not key in AmazHash:
              AmazHash[key]=set()
            AmazHash[key].add(aids[row])
            AmazKey2Hash[aids[row]]=key
        amzn.clear()
        aids.clear()
      
        for i in range(0,len(df)):
          gids.append(df.iloc[i,0])
          if row_size!=901:
            googl.append(df.iloc[i,1:row_size+1])
          else:
            googl.append(df.iloc[i,1:])
          if(len(googl)%64==0):
            b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(googl,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(googl,dtype=np.float64)})[0])
            for row in range(0,len(b1)):
              key=np.array2string(b1[row])
              key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
              if not key in GoogDict:
                GoogDict[key]=1
              else:
                GoogDict[key]+=1
              if not key in GoogHash:
                GoogHash[key]=set()
              GoogHash[key].add(gids[row])
              GoogKey2Hash[gids[row]]=key
            googl.clear()
            gids.clear()
          
        if (len(googl)>0):
          b1= np.sign(_sess.run(batch_outputs1, {states1_ph: np.array(googl,dtype=np.float64), states2_ph: np.array(googl,dtype=np.float64), states3_ph: np.array(googl,dtype=np.float64)})[0])
          for row in range(0,len(b1)):
            key=np.array2string(b1[row])
            key=key.replace('\n', '').replace('\r', '').replace('.', '').replace('-1', '0').replace(' ', '').replace('[', '').replace(']', '')
            if not key in GoogDict:
              GoogDict[key]=1
            else:
              GoogDict[key]+=1
            if not key in GoogHash:
              GoogHash[key]=set()
            GoogHash[key].add(gids[row])
            GoogKey2Hash[gids[row]]=key
        googl.clear()
        gids.clear()
      
        hash_product=0
        for key in list(AmazDict.keys()):
          if key in GoogDict:
            hash_product+=(GoogDict[key]*AmazDict[key])
        print("Computation (%):",100*hash_product/cartesian_product)
        summary["Computation"]=100*hash_product/cartesian_product
        print("Key diversity (Google):",len(GoogDict))
        print("Key diversity (Amazon):",len(AmazDict))
        summary["Key diversity (Google)"]=len(GoogDict)
        summary["Key diversity (Amazon)"]=len(AmazDict)
      
      with open('exec_summary_features_'+str(row_size)+'_code_length_'+str(code_length)+'_whole_data.csv', 'w') as f:
        for key in sorted(summary.keys()):
          f.write("%s,%s\n"%(key,summary[key]))
      
      with open('google_hash_features_'+str(row_size)+'_code_length_'+str(code_length)+'_whole_data.csv', 'w') as f:
        for key in sorted(GoogKey2Hash.keys()):
          f.write("%s,%s\n"%(key,GoogKey2Hash[key]))
      
      with open('amazon_hash_features_'+str(row_size)+'_code_length_'+str(code_length)+'._whole_datacsv', 'w') as f:
        for key in sorted(AmazKey2Hash.keys()):
          f.write("%s,%s\n"%(key,AmazKey2Hash[key]))
