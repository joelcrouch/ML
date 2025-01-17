
import csv
import matplotlib.pyplot as plt
import pandas as pd # Import the library and give a short alias: pd
rent = pd.read_csv("rent-ideal.csv")
print(rent.head(5))

prices = rent['price']
avg_rent = prices.mean()
print(f"Average rent is ${avg_rent:.0f}")


bybaths = rent.groupby(['bathrooms']).mean()
bybaths = bybaths.reset_index() 
print(bybaths[['bathrooms','price']]) 

bybaths.plot.line('bathrooms','price', style='-o')
plt.show()