from sklearn.ensemble import RandomForestClassifier

X=[[1,2,1,16],[2,4,2,32],[3,6,3,64],[4,8,4,128],
   [1,3,2,32],[3,7,4,64],[4,9,5,128],[2,5,3,64]]
y=[0,0,1,2,0,1,2,1]

model=RandomForestClassifier(random_state=42)
model.fit(X,y)

print("Predicted Price Range:",model.predict([[3,7,4,64]])[0])
print("U. Lakshmi Chenna Kesava Reddy - 192425206")