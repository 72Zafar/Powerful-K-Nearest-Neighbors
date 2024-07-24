import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


data = np.array([[25,50000,2],[30,80000,1],[35,60000,3],[20,30000,2],[40,90000,1],[45,75000,2]])
labels = np.array([1,2,1,0,2,1])
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

accuarcy = knn.score(x_test,y_test)*100
print (accuarcy)

user_input = np.array([[35,60000,10]])
user_input_scaled = scaler.transform(user_input)
show = knn.predict(user_input_scaled)
print (show)