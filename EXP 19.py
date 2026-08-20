from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X=[[25,30000,1],[35,50000,1],[45,80000,1],[23,20000,0],
   [30,45000,1],[40,70000,1],[22,18000,0],[28,25000,0]]
y=[1,1,1,0,1,1,0,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("Loan Status:",model.predict([[32,50000,1]])[0])
print("U. Lakshmi Chenna Kesava Reddy - 192425206")