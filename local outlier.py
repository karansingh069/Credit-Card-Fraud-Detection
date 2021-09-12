from common import sns,plt,fraud,xTrain, xTest,yTrain,yTest

print("---------------------------------------------------------------")
print("-------------------------LOCAL OUTLIER ALGORITHM---------------")
print("---------------------------------------------------------------")

from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=25, contamination=0.1)
clf.fit(xTrain,yTrain)
yPred = clf.predict(xTest)
 

from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import confusion_matrix 
n_outliers = len(fraud) 
n_errors = (yPred != yTest).sum() 
print("The model used is Random Forest classifier") 

acc_LOF= accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc_LOF)) 


report = classification_report(yTest,yPred)
print(report)

# printing the confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTest, yPred) 
plt.figure(figsize =(4,4)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, 
			yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 

