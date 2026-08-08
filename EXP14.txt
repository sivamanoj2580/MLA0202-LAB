import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    "Area":[1000,1200,1500,1800,2000,900,1600,1700,2200,1300],
    "Bedrooms":[2,3,3,4,4,2,3,4,5,3],
    "Age":[10,8,5,3,2,12,6,4,1,7],
    "Price":[3000000,4200000,5500000,7000000,8000000,2800000,6000000,6500000,9000000,4500000]
})

X = data[["Area","Bedrooms","Age"]]
y = data["Price"]

model = LinearRegression()
model.fit(X,y)

house = pd.DataFrame({
    "Area":[1600],
    "Bedrooms":[3],
    "Age":[5]
})

price = model.predict(house)

print("Predicted House Price: ₹",round(price[0],2))

print("U.Lakshmi Chenna Kesava Reddy  - 192425206")
