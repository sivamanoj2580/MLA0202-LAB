from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[100,120,140,160,180,200,220,240,260,280]

model=LinearRegression()
model.fit(X,y)

print("R2 Score:",r2_score(y,model.predict(X)))
print("Future Sales:",model.predict([[11]])[0])
print("U. Lakshmi Chenna Kesava Reddy - 192425206")