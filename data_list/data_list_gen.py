import numpy as np  ##引入numpy模块，主要是处理矩阵的函数，定义为np
"""
Train_list = np.loadtxt("./OCT_XRay_train.txt", dtype=str)

Test_list = np.loadtxt("./OCT_XRay_test.txt", dtype=str)


print(len(Train_list),len(Test_list))


Train_list = np.concatenate([Train_list,Test_list],0)
print(len(Train_list),len(Test_list))

permutation = np.random.permutation(Train_list.shape[0])####数据随机
train_dataset = Train_list[permutation]

f = open("./OCT.txt","w")
for i in range(len(train_dataset)):
    if  "DRUSEN" in train_dataset[i]:
         print(train_dataset[i],0)
         f.write(train_dataset[i]+"  "+str(0)+"\n")
    if  "NORMAL" in train_dataset[i]:
         print(train_dataset[i],1)
         f.write(train_dataset[i]+"  "+str(1)+"\n")
    if  "CNV" in train_dataset[i]:
         print(train_dataset[i],2)
         f.write(train_dataset[i]+"  "+str(2)+"\n")
    if  "DME" in train_dataset[i]:
         print(train_dataset[i],3)
         f.write(train_dataset[i]+"  "+str(3)+"\n")
f.close()
"""
import xlrd



Train_list = np.loadtxt("./Messidor.txt", dtype=str)
list11 = np.loadtxt("/home/xly/data/Messidor/Base11/Annotation.txt", dtype=str, delimiter='  ')

print(len(Train_list),list11.shape)


# permutation = np.random.permutation(Train_list.shape[0])####数据随机
# train_dataset = Train_list[permutation]

f = open("./Messidor_4.txt","w")
for i in range(len(list11)):
    for j in Train_list:
        if list11[i][0] in j:
           f.write(j+"  "+str(list11[i][1])+"\n")
           print(list11[i],j)

f.close()