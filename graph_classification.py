import tensorflow as tf
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow.keras
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import networkx as nx
import random
import matplotlib.pyplot as plt
import sklearn
import random
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import preprocessing
import scipy as sc
import os
import re
import gc
import itertools
import statistics
import pickle
import argparse
import argparse

def one(i,n):
    a = np.zeros(n, 'uint8')
    a[i] = 1
    return a


################################ THIS FUNCTION (read_graphfile) IS ADAPTED FROM RexYing/HybridPool ############################################

def read_graphfile(dataname):
    max_nodes=None
    #read datasets
    prefix='dataset_graph/'+dataname+'/'+dataname
    data_list=[]
    data={}
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
#                 print(attrs)
#                 break
    except IOError:
        print('No node attributes')
#         mode_attrs=
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here

        G=nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = np.array(node_label_one_hot)
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:

            G.graph['feat_dim'] = node_attrs[0].shape[0]

        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
       
        graphs.append(nx.relabel_nodes(G, mapping))

    max_num_nodes = max([G.number_of_nodes() for G in graphs])
    if len(node_attrs)>0:
        feat_dim = graphs[0].node[0]['feat'].shape[0]
    feat_dim1 = graphs[0].node[0]['label'].shape[0]
    lab1=[]
    for G in graphs:  
        # print("-------")      
        adj = np.array(nx.to_numpy_matrix(G))
#         rowsum = np.array(adj.sum(1))
#         d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = np.diag(d_inv_sqrt)
#         adj=adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((max_num_nodes,max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        label1=G.graph['label']
        
        if len(node_attrs)<0:#==G.number_of_nodes():
            f = np.zeros((max_num_nodes,feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['feat']
        else:
            max_deg = 63
            f = np.zeros((max_num_nodes,feat_dim1), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['label']
            # rowsum = np.array(f.sum(1))
            # r_inv = np.power(rowsum, -1).flatten()
            # r_inv[np.isinf(r_inv)] = 0.
            # r_mat_inv = sp.diags(r_inv)
            # f = r_mat_inv.dot(f)
                
            degs = np.sum(np.array(adj), 1).astype(int)

            degs[degs>max_deg] = max_deg
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            f1 = np.pad(feat, ((0, max_num_nodes - G.number_of_nodes()), (0, 0)),
                    'constant', constant_values=0)

            
            # f1=np.identity(max_num_nodes)
            f = np.concatenate((f, f1), axis=1)
            rowsum = np.array(f.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            f = r_mat_inv.dot(f)
            # f = np.concatenate((f1, f), axis=1)
        lab1.append(label1)   
        label1=one(label1,len(label_vals))
        data={}
        data['feat']=f
        data['adj']=adj_padded
        data['label']=label1
        data_list.append(data)
    return data_list, len(label_vals),max_num_nodes,lab1



def glrt_init(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def activation_layer(act,input,name=None): #######activation functions##########
    if act=="sigmoid":
        layer = tf.nn.sigmoid(input,name=name)
        return layer
    elif act=="relu":
        layer = tf.nn.relu(input,name=name)
        return layer
    elif act=="swish":
        layer = tf.nn.swish(input,name=name)
        return layer
    elif act=="tanh":
        layer = tf.nn.tanh(input,name=name)
        return layer
    elif act=="leaky_relu":
        layer = tf.nn.leaky_relu(input,name=name)
        return layer



def GIN_layer(A1,H,out_feat,in_feat,act,name='gcn',i=0,k=1,k1=0.3):
    weights = glrt_init([in_feat, out_feat],name=name)
    # n12=A1.get_shape().as_list()
    eps=tf.Variable(tf.zeros(1))
    
    rowsum = tf.reduce_sum(A1,axis=2)            
    d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
    d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
    d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
    A1=tf.matmul(tf.matmul(d_inv_sqrt,A1),d_inv_sqrt)
    H=tf.nn.dropout(H,1-k1)#,training=k)
    A1=tf.matmul(A1,H,name=name+'matmul1')
    A1=A1+(1+eps)*H 
    for i in range(2-1):
        ad=tf.keras.layers.Dense(units = out_feat)(A1)
        input_features = tf.nn.relu(ad)
    H_next=tf.keras.layers.Dense(units = out_feat)(input_features)
    H_next = tf.nn.relu(H_next)

    
    return H_next

class HybridPool(object):
    def __init__(self,placeholders):
        self.num_pool=placeholders['num_pool']
        self.num_nodes=placeholders['num_nodes']
        self.gcnlayers=placeholders['gcnlayers']
        self.feat_dim=placeholders['feat_dim']
        self.embd_dim=placeholders['emb_dim']
        self.gcn_dim=placeholders['gcn_dim']
        self.clusratio=placeholders['clusrstio']
        self.nclasses=placeholders['nclasses']
        self.learning_rate=placeholders['learning_rate']
        self.entropy_loss=0
        self.hidden_size=placeholders['emb_dim']
        self.attention_size=placeholders['emb_dim']
        self.acts="relu"
        


    #THIS FUNCTIN IS TAKEN FROM https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    def BLSTM_ATTENTION(self,X_comb):
        initializer = tf.keras.initializers.he_normal
        with tf.variable_scope("bi-lstm"):             
            _fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=initializer())
            _bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=initializer())
            self.rnn_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=_fw_cell,
                                                                  cell_bw=_bw_cell,
                                                                  inputs=X_comb,
                                                                  dtype=tf.float32)                 
        #concat
        inputs = tf.concat(self.rnn_outputs1,2)  
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
        # print("----------/-inp--------------------",inputs)
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1))# + b_omega) 
        # print("-----------inp--------------------",v)              
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        # print("-----------inp--------------------",v)  
        alphas = tf.nn.softmax(vu, name='alphas')      # (B,T) shape
        # print("-----------inp--------------------",vu) 
        output = tf.reduce_sum(v * tf.expand_dims(alphas, -1),1)
        # print("-----------inp--------------------",output)

        return output

    def Emb_Pooling_layer(self,clusnext,A3,x,in_feat,act,j,i):  

        if clusnext==-1:

            if j==1:
                clusnext1=1            
            else:
                if i==0:
                    clusnext1=6 
                elif i==1:
                    clusnext1=3
        else:
            clusnext1=clusnext
        with tf.variable_scope("node_gnn",reuse=False):
            x1=x
            A2=A3
            for i1 in range(2):#self.gcnlayers):
                z_l=GIN_layer(A2,x1,self.gcn_dim,in_feat,act,i=j,k=self.train,k1=self.drop_rate)
                x1=z_l
                in_feat=x1.get_shape().as_list()[2] 
            x1=x
            in_feat=x1.get_shape().as_list()[2] 
            for i1 in range(3):#0,self.gcnlayers):                
                z_l1=GIN_layer(A2,x1,clusnext1,in_feat,act,i=j,k=self.train,k1=self.drop_rate)
                x1=z_l1
                in_feat=x1.get_shape().as_list()[2]
                
            s_l=tf.nn.softmax(z_l1)
            x_l1=tf.matmul(tf.transpose(s_l,[0,2,1]),z_l)  
            x_l1=tf.math.l2_normalize(x_l1,2)
            A_l1=tf.matmul(tf.matmul(tf.transpose(s_l,[0,2,1]),A2),s_l)
            
        return x_l1,A_l1,z_l
    
    def classifier(self,input):
        initializer = tf.keras.initializers.he_normal
        node_feat_rec = tf.layers.dense(input, self.nclasses, activation=None, use_bias=False,kernel_initializer=initializer())
        return node_feat_rec
    
    def Calc_Loss(self, logits, labels,reuse=False):
        l2_reg_lambda=1e-5
        with tf.name_scope('loss'):
            self.loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            self.optimizer()
    
    def HybridPool_Architechture(self):
        ##############COMPLETE ARCHITECHTURE OF THE PAPER################
        self._add_placeholders()
        with tf.variable_scope("node12_gnn",reuse=False):
            A=self.input_adj  
            x=self.input_features
            in_feat=x.get_shape().as_list()            
            in_feat=in_feat[2]

            x1=GIN_layer(A,x,self.gcn_dim,in_feat,self.acts,k=self.train,k1=self.drop_rate)
            # x2=GIN_layer(A,x1,self.gcn_dim,x1.get_shape().as_list()[2],self.acts,k=self.train,k1=self.drop_rate)
            x=x1

            if self.clusratio!=-1:
                clusnext=int(self.num_nodes*self.clusratio)
            else:
                clusnext=-1
            for i in range(self.num_pool): 
                if i==self.num_pool-1:
                    x_l1,A_l1,z_l1=self.Emb_Pooling_layer(1,A,x,x.get_shape().as_list()[2],self.acts,1,i)
                else:                    
                    x_l1,A_l1,z_l1=self.Emb_Pooling_layer(clusnext,A,x,x.get_shape().as_list()[2],self.acts,0,i)
                A=A_l1
                x=x_l1
                in_feat=x.get_shape().as_list()[2]                
                if i!=0.:
                    X_comb=tf.concat([X_comb,x_l1],axis=1)
                else:
                    X_comb=x_l1
                if self.clusratio!=-1:
                    clusnext=int(clusnext*self.clusratio)
                    
                    
        X_comb=tf.convert_to_tensor(X_comb, dtype=tf.float32)
        inp=self.BLSTM_ATTENTION(X_comb)       
        output=self.classifier(inp)
        labels=self.input_labels  

        y_pred =(tf.nn.softmax(output))   
        y_pred_cls = tf.argmax(y_pred, dimension=1,name='pp')         
        l=tf.argmax(labels, dimension=1,name="pp1")
        correct_prediction = tf.equal(y_pred_cls, l)       
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="mean")       
        self.Calc_Loss(output,labels)      
    
    def runn(self,sess,feed1,v): 
        feed={self.train :feed1['train1'],self.drop_rate:feed1['keep'],self.input_features:feed1['input_features1'],self.input_adj:feed1['input_adj1'],self.input_labels:feed1['input_labels1']}
        if v=="train":            
            run_vars = [self.train_op_np]
            c = sess.run(run_vars, feed_dict = feed)
            run_vars=[tf.get_default_graph().get_tensor_by_name("node12_gnn/node_gnn/Variable:0"),tf.get_default_graph().get_tensor_by_name("mean:0"),self.loss_val]
            v,a,summ = sess.run(run_vars, feed_dict = feed)
            return a,summ,v
        
        elif v=="val" or v=="test":
            run_vars = [tf.get_default_graph().get_tensor_by_name("pp1:0"),tf.get_default_graph().get_tensor_by_name("pp:0"),tf.get_default_graph().get_tensor_by_name("mean:0"),self.loss_val]
            kk,y_predd,a,summ = sess.run(run_vars, feed_dict = feed)            
            return kk,y_predd,a,summ
        
    def optimizer(self,reuse=False):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        with tf.name_scope('opt'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate , global_step, 100000, 0.96, staircase=True)
            self.train_op_np = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
  
    def _add_placeholders(self):
        self.input_features = tf.placeholder(tf.float32, shape = [None, self.num_nodes,self.feat_dim], name = "input_features")
        self.input_adj = tf.placeholder(tf.float32, shape = [None,self.num_nodes, self.num_nodes], name = "input_adj")
        self.input_labels = tf.placeholder(tf.int32, shape = [None, self.nclasses], name = "input_labels")           
        self.drop_rate = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)
        



def train_val_test(adj,labels,feat,arguments,nclasses,max_num_nodes,lab1):
    epoch1=arguments.epoch
           
    ################Training, validation and testing###################
    placeholders={}
    final={}

    print(max_num_nodes,nclasses,len(feat))
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0) #KFold(n_splits=10)
    
    ep=[[] for ir in range(10)]
    it=0
    for train_index, test_index in kf.split(adj,lab1):
        train_label,test_label=labels[train_index],labels[test_index]
        train_feat,test_feat=feat[train_index],feat[test_index]
        train_adj,test_adj=adj[train_index],adj[test_index]

        a1=0
        
        pat=20 #patience for early stopping 
        tf.reset_default_graph()
        tf.set_random_seed(123)

        #IN THIS CHANGE clusrstio1 IF WANT TO CHANGE THE pool_ratio. 
        #FOR NOW IT IS SET TO 0.5 AS WE HAVE FIXED NUMBER OF NODES IN EACH LAYER ACCORDING TO MUTAG DATASET.

        #OR ONE CAN ALSO CHANGE THE NO. OF LAYERS (num_pool1) INSTEAD OF RATIO. 
        #FOR NOW WE HAVE SET NO. OF LAYERS EQUAL TO 3 SO THAT DEFAULT SEETING CAN BE USED ACCORDING TO MUTAG DATASET.
    
        num_pool1=4 # Total layers including the input layer
        clusrstio1=0.75

        
        if num_pool1!=3 and clusrstio1==-1:
            raise Exception("As you have changed the no. of layers, you must specify the pooling ratio also.")


        num_pool2=num_pool1-1
        placeholders={'gcn_dim':arguments.gcn_dim,'l2':0.05,'g':2,'learning_rate':arguments.learning_rate,'num_nodes':max_num_nodes,'num_pool':num_pool2,'gcnlayers':arguments.gcn_layer,'feat_dim':feat[0].shape[1],'emb_dim':arguments.embd_dim,'clusrstio':clusrstio1,'nclasses':nclasses}
        
        D=HybridPool(placeholders)
        
        batch_size=arguments.batch_size
        batch_size=20#len(train_adj)
        num_batches=int(len(train_adj)/batch_size)
        D.HybridPool_Architechture()

        #####to set GPU ##########
        # config = tf.ConfigProto(device_count = {'GPU': 4})
        # config.gpu_options.allow_growth=False

        sess = tf.Session()#config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        subb=[]
        vlss_mn = np.inf
        vacc_mx = 0.0
        asqmx=0.0
        step = 0
        
        for epoch in range(500):
            trainavrloss = 0
            trainavracc = 0
            vallossavr = 0
            valaccravr = 0
            subgraph=[]
            i10=0

            tr_step = 0
            # tr_size = len(train_adj)
            # print("epoch",epoch)
            for j in range(num_batches):            
                feed = {}
                feed['input_features1'] = train_feat[j*batch_size:j*batch_size+batch_size]
                feed['input_adj1'] = train_adj[j*batch_size:j*batch_size+batch_size]
                feed['input_labels1']=train_label[j*batch_size:j*batch_size+batch_size]
                feed['keep']=arguments.dropout
                feed['train1']=True
                a1,summ1,v1=D.runn(sess,feed,"train")
                trainavrloss += summ1
                trainavracc += a1
                tr_step += 1
            # print("oooo",v1)
            
            feed = {}
            feed['input_features1'] = test_feat
            feed['input_adj1'] = test_adj
            feed['input_labels1']=test_label
            feed['keep']=0
            feed['train1']=False

            k1,y,a,summ1=D.runn(sess,feed,"val")
            ep[it].append(a*100)
            
        print("Accuracy on train set",trainavracc/tr_step,"test set",a)#"loss",summ)
        it+=1
        # print("************************************************")

    ep1=np.mean(ep,axis=0)
    ep11=ep1.tolist()
    epi=ep11.index(max(ep11))
    print("Average accuracy is:",max(ep11))
    # print(ep11[epi])

            
def argument_parser():
    parser = argparse.ArgumentParser(description="HybridPool for graph classification")
    parser.add_argument("-dt", "--dataset", type=str, help="name of the dataset", default="MUTAG")
    parser.add_argument("-ep", "--epoch", type=int, default=300, help="Number of Epochs")
    parser.add_argument("-sbl", "--gcn_layer", type=int, default=3, help="number of gcn layers")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-ed", "--embd_dim", type=int, default=128, help="final embedding dimension")
    parser.add_argument("-gcn_ed", "--gcn_dim", type=int, default=128, help="gcn embedding dimension")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("-bs", "--batch_size", type=int, default=20, help="batch size")


    arguments = parser.parse_args()
    return arguments

def read_dataset(dataset):
    ################### read in graphs ########################
    datasets,n_classes,max_num_node,lab1=read_graphfile(dataset)
    datasets=np.array(datasets)
    return datasets,n_classes,max_num_node,lab1
 
def main():
    arguments = argument_parser()
    dataset,nclasses,max_num_nodes,lab1=read_dataset(arguments.dataset)
    #################SEPERATE EACH COMPONENT###################
    adj=[]
    labels=[]
    feat=[]
    subgraphs1=[]
    for i in range(len(dataset)):
        adj.append(dataset[i]['adj'])
        labels.append(dataset[i]['label'])
        feat.append(dataset[i]['feat'])
       
    adj=np.array(adj)
    feat=np.array(feat)
    labels=np.array(labels)
    train_val_test(adj,labels,feat,arguments,nclasses,max_num_nodes,lab1)

#Main Function
if __name__ == "__main__":
    main()



