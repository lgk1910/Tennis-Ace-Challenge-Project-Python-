import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
pd.options.display.max_columns = 100

# load and investigate the data here:
tennis_stats = pd.read_csv('VSCode\Python\Codecademy\Challenges\\tennis_ace_starting\\tennis_ace_starting\\tennis_stats.csv')
# print(tennis_stats.head())




# perform exploratory analysis here:
# for column in tennis_stats.columns:
#     plt.scatter(tennis_stats.Winnings,tennis_stats[column])
#     plt.title('Winings vs {}'.format(column))
#     plt.show()




## perform single feature linear regressions here:
print('Linear Regression')
single_feat_model = LinearRegression()
x_values = np.array(tennis_stats.BreakPointsOpportunities).reshape(-1,1)
y_values = tennis_stats.Winnings
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
single_feat_model.fit(x_train, y_train)
print('Train score: '+str(single_feat_model.score(x_train, y_train)))
print('Test score: '+str(single_feat_model.score(x_test, y_test)))
y_predicted = single_feat_model.predict(x_test)
plt.scatter(y_test, y_predicted, alpha = 0.3)
plt.xlabel('Reality')
plt.ylabel('Prediction')
plt.title('Reality vs Prediction')
# plt.show()

# plt.clf()
# print('Linear Regression')
# single_feat_model_2 = LinearRegression()
# x_values = np.array(tennis_stats.Wins).reshape(-1,1)
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# single_feat_model_2.fit(x_train, y_train)
# print('Train score: '+str(single_feat_model_2.score(x_train, y_train)))
# print('Test score: '+str(single_feat_model_2.score(x_test, y_test)))
# y_predicted = single_feat_model_2.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# plt.show()

# plt.clf()
# single_feat_model_3 = LinearRegression()
# x_values = np.array(tennis_stats.ReturnGamesPlayed).reshape(-1,1)
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# single_feat_model_3.fit(x_train, y_train)
# print('Train score: '+str(single_feat_model_3.score(x_train, y_train)))
# print('Test score: '+str(single_feat_model_3.score(x_test, y_test)))
# y_predicted = single_feat_model_3.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# plt.show()

# plt.clf()
# single_feat_model_4 = LinearRegression()
# x_values = np.array(tennis_stats.Aces).reshape(-1,1)
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# single_feat_model_4.fit(x_train, y_train)
# print('Train score: '+str(single_feat_model_4.score(x_train, y_train)))
# print('Test score: '+str(single_feat_model_4.score(x_test, y_test)))
# y_predicted = single_feat_model_4.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# plt.show()

# plt.clf()
# single_feat_model_5 = LinearRegression()
# x_values = np.array(tennis_stats.ServiceGamesPlayed).reshape(-1,1)
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# single_feat_model_5.fit(x_train, y_train)
# print('Train score: '+str(single_feat_model_5.score(x_train, y_train)))
# print('Test score: '+str(single_feat_model_5.score(x_test, y_test)))
# y_predicted = single_feat_model_5.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# plt.show()



## perform two feature linear regressions here:


# plt.clf()
# print('Two Feature Linear Regression:')
# double_feat_model = LinearRegression()
# x_values = tennis_stats[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# double_feat_model.fit(x_train, y_train)
# print('Train score: '+str(double_feat_model.score(x_train, y_train)))
# print('Test score: '+str(double_feat_model.score(x_test, y_test)))
# y_predicted = double_feat_model.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# # plt.show()

# plt.clf()
# print('Two Feature Linear Regression:')
# double_feat_model = LinearRegression()
# x_values = tennis_stats[['ReturnGamesPlayed', 'ServiceGamesPlayed']]
# y_values = tennis_stats.Winnings
# x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
# double_feat_model.fit(x_train, y_train)
# print('Train score: '+str(double_feat_model.score(x_train, y_train)))
# print('Test score: '+str(double_feat_model.score(x_test, y_test)))
# y_predicted = double_feat_model.predict(x_test)
# plt.scatter(y_test, y_predicted, alpha = 0.3)
# plt.xlabel('Reality')
# plt.ylabel('Prediction')
# plt.title('Reality vs Prediction')
# plt.show()

plt.clf()
print('Two Feature Linear Regression:')
double_feat_model = LinearRegression()
x_values = tennis_stats[['BreakPointsOpportunities', 'Aces']]
y_values = tennis_stats.Winnings
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
double_feat_model.fit(x_train, y_train)
print('Train score: '+str(double_feat_model.score(x_train, y_train)))
print('Test score: '+str(double_feat_model.score(x_test, y_test)))
y_predicted = double_feat_model.predict(x_test)
plt.scatter(y_test, y_predicted, alpha = 0.3)
plt.xlabel('Reality')
plt.ylabel('Prediction')
plt.title('Reality vs Prediction')
# plt.show()


## perform multiple feature linear regressions here:
plt.clf()
print('Multiple Feature Linear Regression:')
multiple_feat_model = LinearRegression()
x_values = tennis_stats[['BreakPointsOpportunities', 'Aces', 'BreakPointsFaced', 'ServiceGamesPlayed','ReturnGamesPlayed']]
# x_values = tennis_stats[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
# 'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
# 'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
# 'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
# 'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
# 'TotalServicePointsWon']]
y_values = tennis_stats.Winnings
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.8, test_size = 0.2)
multiple_feat_model.fit(x_train, y_train)
print('Train score: '+str(multiple_feat_model.score(x_train, y_train)))
print('Test score: '+str(multiple_feat_model.score(x_test, y_test)))
y_predicted = multiple_feat_model.predict(x_test)
plt.scatter(y_test, y_predicted, alpha = 0.3)
plt.xlabel('Reality')
plt.ylabel('Prediction')
plt.title('Reality vs Prediction')
plt.show()
