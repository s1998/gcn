import pickle
import operator
import numpy as np
import csv
import os.path

with open ('y_test', 'rb') as f:
	y_test=pickle.load(f)

dicvocab={}
f=open("data/vocab.csv")
vocab=csv.reader(f)
for word in vocab:
    if word[0]!='':
        dicvocab[int(word[0])-1]=word[1]
f.close()

label_size=y_test.shape[1]

topics=["/Artificial_Intelligence/Machine_Learning/Case-Based/", "/Artificial_Intelligence/Machine_Learning/Genetic_Algorithms/", "/Artificial_Intelligence/Machine_Learning/Neural_Networks/", "/Artificial_Intelligence/Machine_Learning/Probabilistic_Methods/", "/Artificial_Intelligence/Machine_Learning/Reinforcement_Learning/", "/Artificial_Intelligence/Machine_Learning/Rule_Learning/", "/Artificial_Intelligence/Machine_Learning/Theory/"]
maxlabel=np.argmax(y_test, axis=1)
for ind in range(label_size+1):
	st='dictionary' + str(ind)
	if not os.path.isfile(st):
		continue
	if ind<label_size: print("Class ",ind, "enabled")
	else: print("All Classes enabled") 
	with open(st, 'rb') as f:
		dic= pickle.load(f)
	for i in range(label_size):
		dic2={}
		print("Top 5 Highest Relevance Features for Class ", topics[i], "->", end='')
		for x in dic[i]:
			if x not in dic2: dic2[x]=0
			dic2[x]+=1
		k=sorted(dic2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		for z in k[:15]:
			print((dicvocab[z[0]],z[1]),end=',')
		print()
	print()
	print()