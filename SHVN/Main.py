#Required imports
import torch
from Teacher import Teacher
from Model import Model
from data import load_data,NoisyDataset
from util import accuracy
from Student import Student

class Arguments():
    
#Class used to set hyperparameters for the whole PATE implementation
    def __init__(self):
        
        self.batchsize = 64
        self.test_batchsize = 10
        self.epochs=50
        self.student_epochs=10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.n_teachers=50
        self.save_model = False
        

args = Arguments()

train_loader = load_data("train",args.batchsize,"SHVN")
test_loader=load_data("test",args.test_batchsize,"SHVN")

#Declare and train teachers on MNIST training data
teacher=Teacher(args,Model,n_teachers=args.n_teachers)
#teacher.load_models()
teacher.train(train_loader)
#teacher.save_models()
#Label public unlabelled data

targets=[]
predict=[]

for data,target in test_loader:
    
    targets.append(target)
    predict.append(teacher.predict(data))
    
print("Accuracy: ",accuracy(torch.tensor(predict),targets))

print("\n")
print("\n")

print("Training Student")

print("\n")
print("\n")

student=Student(args,Model())
N=NoisyDataset(train_loader,teacher.predict)
student.train(N)

results=[]
targets=[]

total=0.0
correct=0.0

for data,target in test_loader:
    
    predict_lol=student.predict(data)
    correct += float((predict_lol == (target)).sum().item())
    total+=float(target.size(0))
    
print((correct/total)*100)

    
#print("Student Accuracy: ",accuracy(results,targets))
#print("Real accuracy",accuracy(torch.tensor(real_results),test_targets))
#print("Accuracy: ",accuracy(torch.tensor(results),test_targets))

