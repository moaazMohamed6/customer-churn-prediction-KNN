from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("telecom_churn_clean.csv")

x = df[["account_length","total_day_minutes","customer_service_calls"]]
y = df[["churn"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42,stratify=y)


train_accurecy = {}
test_accurecy = {}

for i in range (1,21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    train_accurecy[i] = knn.score(x_train,y_train)
    test_accurecy[i] = knn.score(x_test,y_test)
    score = 0
    for s in test_accurecy.values():
        if s > score :
            score = s
print(score)            


plt.plot(range(1,21), list(train_accurecy.values()), label='Train Accuracy')
plt.plot(range(1,21), list(test_accurecy.values()), label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

    
