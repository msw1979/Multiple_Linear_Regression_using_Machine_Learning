# Author Dr. M. Alwarawrah
import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# start recording time
t_initial = time.time()

#Columns names
col_names = ["Make","Model","Vehicle_Class","Engine_Size","Cylinders","Transmission","Fuel_Type","Fuel_Consumption_City","Fuel_Consumption_Hwy","Fuel_Consumption_Comb","Fuel_Consumption_Comb_mpg","CO2_Emissions"]
#Read dataframe and skip first raw that contain header
df = pd.read_csv('CO2 Emissions_Canada.csv',names=col_names, header = None, skiprows = 1)

#print Dataframe information
#print(df.describe())

#draw histograms for the following features
plt.clf()
cdf = df[["Engine_Size","Cylinders","Fuel_Consumption_Comb","CO2_Emissions"]]
cdf.hist()
plt.savefig("viz_hist.png")

# plot scatter plots
plt.clf()
fig, ax = plt.subplots(1,3)
ax[0].scatter(cdf.Engine_Size, cdf.CO2_Emissions,  color='k')
ax[0].set_xlabel("Engine Size") 
ax[0].set_ylabel("CO2 Emissions")
ax[1].scatter(cdf.Cylinders, cdf.CO2_Emissions,  color='k')
ax[1].set_xlabel("Cylinders") 
ax[1].set_ylabel("CO2 Emissions")
ax[2].scatter(cdf.Fuel_Consumption_Comb, cdf.CO2_Emissions,  color='k')
ax[2].set_xlabel("Fuel Consumption Comb") 
ax[2].set_ylabel("CO2 Emissions")
fig.tight_layout()
plt.savefig("scatter.png")

#select data for train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# plot scatter plots and show train (blue) and test data (red)
plt.clf()
fig, ax = plt.subplots(3,1)
ax[0].scatter(train.Engine_Size, train.CO2_Emissions,  color='b', label='Train', marker="x", s=5)
ax[0].scatter(test.Engine_Size, test.CO2_Emissions,  color='r', label='Test', marker=".", s=5)
ax[0].set_xlabel("Engine Size") 
ax[0].set_ylabel("CO2 Emissions")
ax[0].legend(loc='best',frameon=False,fontsize = "8")
ax[1].scatter(train.Cylinders, train.CO2_Emissions,  color='b', label='Train', marker="x", s=5)
ax[1].scatter(test.Cylinders, test.CO2_Emissions,  color='r', label='Test', marker=".", s=5)
ax[1].set_xlabel("Cylinders") 
ax[1].set_ylabel("CO2 Emissions")
ax[1].legend(loc='best',frameon=False,fontsize = "8")
ax[2].scatter(train.Fuel_Consumption_Comb, train.CO2_Emissions,  color='b', label='Train', marker="x", s=5)
ax[2].scatter(test.Fuel_Consumption_Comb, test.CO2_Emissions,  color='r', label='Test', marker=".", s=5)
ax[2].set_xlabel("Fuel Consumption Comb") 
ax[2].set_ylabel("CO2 Emissions")
ax[2].legend(loc='best',frameon=False,fontsize = "8")

fig.tight_layout()
plt.savefig("scatter_train_test.png")

#create a file to write fit information and regression accuracy
output_file = open('linear_reg_output.txt','w')

# This function use apply multiple linear regression using ML
# You need to provide the train and test data set, columns names, and outfile name 

#define linear regression
regr = linear_model.LinearRegression()
#define train data for x and y
train_x = np.asanyarray(train[["Engine_Size","Cylinders","Fuel_Consumption_Comb"]])
train_y = np.asanyarray(train[["CO2_Emissions"]])
# apply the linear regression to the x & y train data
regr.fit(train_x, train_y)

print ('Multiple linear regression for CO2 Emissions vs. Engine Size, Cylinders and Fuel Consumption Comb', file=output_file)

# The coefficients
for i in range(0,len(train_x[0])):
    print ('a%d: %.2f'%(i,regr.coef_[0][i]), file=output_file)

#define the test data for x and y
test_x = np.asanyarray(test[["Engine_Size","Cylinders","Fuel_Consumption_Comb"]])
test_y = np.asanyarray(test[["CO2_Emissions"]])
# find the prediction for y
y_predict = regr.predict(test_x)

# Regression accuracy
MAE = np.mean(np.absolute(test_y-y_predict))
print("Mean Absolute Error (MAE): %.2f" % MAE, file=output_file)
MSE = np.mean((test_y-y_predict) ** 2)
print("Mean Square Error (MSE): %.2f" % MSE, file=output_file)
RMSE = np.sqrt(MSE)
print("Root Mean Square Error (RMSE): %.2f" % RMSE, file=output_file)
RAE = np.sum(np.absolute(test_y-y_predict))/np.sum(np.absolute(test_y-np.mean(test_y)))
print("Relative Absolute Error (RAE): %.2f" % RAE, file=output_file)
RSE = np.sum((test_y-y_predict)**2)/np.sum((test_y-np.mean(test_y))**2)
print("Relative Square Error (RSE): %.2f" % RSE, file=output_file)
#R2 = r2_score(test_y , y_predict) 
#print("R2-score: %.2f" %R2, file=output_file)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_x, test_y),file=output_file)

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))