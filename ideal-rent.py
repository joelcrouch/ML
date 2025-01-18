
import csv
import matplotlib.pyplot as plt
import pandas as pd # Import the library and give a short alias: pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rent = pd.read_csv("rent-ideal.csv")
print(rent.head(5))

prices = rent['price']
avg_rent = prices.mean()
print(f"Average rent is ${avg_rent:.0f}")


bybaths = rent.groupby(['bathrooms']).mean()
bybaths = bybaths.reset_index() 
print(bybaths[['bathrooms','price']]) 

# bybaths.plot.line('bathrooms','price', style='-o')
# plt.show()

X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
print(type(X), type(y))

rf=RandomForestRegressor(n_estimators=10)  #use 10 trees
rf.fit(X,y)

unknown_x = [2, 1, 40.7957, -73.97] # 2 bedrooms, 1 bathroom, ...
predicted_y = rf.predict([unknown_x])
print(predicted_y)

predictions=rf.predict(X)
e=mean_absolute_error(y, predictions) 
ep=e*100.0/y.mean()
print(f"${e:.0f} average error; {ep:.2f}% error")

X, y = rent[['latitude','longitude']], rent['price']
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
location_e = mean_absolute_error(y, rf.predict(X))
location_ep = location_e*100.0/y.mean()
print(f"${location_e:.0f} average error; {location_ep:.2f}% error")