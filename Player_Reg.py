import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import tree
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv('IPL IMB381IPL2013.csv')
pd.set_option('display.max_rows', 130)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 500)


#print(data.info())

x_features=['AGE','COUNTRY','PLAYING ROLE','T-RUNS','T-WKTS','ODI-RUNS-S','ODI-SR-B',
            'ODI-WKTS','ODI-SR-BL','CAPTAINCY EXP','RUNS-S','HS','AVE','SR-B','SIXERS',
            'RUNS-C','WKTS','AVE-BL','ECON','SR-BL']
#print(Features.shape)


#categorical to numeric

cate_features=['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP']
player_encoded_df=pd.get_dummies(data[x_features],columns=cate_features,drop_first=True)
#print(player_encoded_df)
Featuers=player_encoded_df
y=data['SOLD PRICE']
#Variance of features

def get_vif_factors(x):
    x_matrix=x.to_numpy()
    vif=[variance_inflation_factor(x_matrix,i) for i in range (x_matrix.shape[1])]
    vif_factors=pd.DataFrame()
    vif_factors['column']=x.columns
    vif_factors['VIF']=vif
    vif_factors=vif_factors.sort_values(by='VIF',ascending=False)
    return vif_factors
#print(get_vif_factors(x_features))
vif_factors=get_vif_factors(Featuers)
columns_with_large_vif = vif_factors[vif_factors['VIF'] > 4].column
#print(vif_factors)


#the bigest value of VIF that will be removed

#print(columns_with_large_vif.shape)
#print(columns_with_large_vif)

#Visualize the corr of featuers
plt.figure(figsize=(12,10))
sns.heatmap(Featuers[columns_with_large_vif].corr(), annot = True)
plt.title("heatmap depicting corr between features")
#plt.show()

#remove multicollinear features
columns_to_be_removed = ['T-RUNS','T-WKTS','RUNS-S', 'HS', 'AVE','RUNS-C','SR-B','AVE-BL',
                            'ECON', 'ODI-SR-B','ODI-RUNS-S','SR-BL', 'AGE_2']
x_new_features=list(set(Featuers)-set(columns_to_be_removed))
#print(np.shape(x_new_features))
#print(get_vif_factors(Featuers[x_new_features]))

X_train, X_test, Y_train, Y_test = train_test_split(Featuers, y, test_size=0.8, random_state=42)
model=sm.OLS(Y_train,X_train).fit()

X_train = X_train[x_new_features]
model_2=sm.OLS(Y_train,X_train).fit()
print(model_2.summary())

significant_vars=['COUNTRY_ENG','COUNTRY_IND','SIXERS','CAPTAINCY EXP_1']
X_train, X_test, Y_train, Y_test = train_test_split(Featuers[significant_vars], y, test_size=0.8, random_state=42)
model_3=LinearRegression().fit(X_train,Y_train)
y_pre=model_3.predict(X_test)
pred_train_lr=model_3.predict(X_train)
#print(model_3.summary())
print("mean_squared_error(Train): ",np.sqrt(mean_squared_error(Y_train,pred_train_lr)))
print("r2_score(Train): ",r2_score(Y_train,pred_train_lr))
print("mean_squared_error(Test): ",np.sqrt(mean_squared_error(Y_test,y_pre)))
print("r2_score(Test): ",r2_score(Y_test,y_pre))