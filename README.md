# Understanding-soccer-scouting-using-ML.-
An algorithm that can decompose the performance of players into its main objective elements using scouting data as a strategic asset.By analyzing large datasets of player performance metrics, biographical information, and other relevant data, our algorithm will identify the key performance metrics that contribute to a player's overall success, such as physical attributes, technical skills, and tactical awareness.

By analyzing large datasets of player performance metrics, biographical information, and other relevant data, our algorithm will identify the key performance metrics that contribute to a player's overall success, such as physical attributes, technical skills, and tactical awareness.

## Dataset
The scouting dataset has about 20,000 data points, each representing a player, their
team, and their performance metrics.
The target variable for this dataset is "rating_num," which represents the player's
overall rating. This rating is a continuous variable ranging from 0 to 10.There are
several features included in this dataset that provide information about the player's
physical attributes, playing position, and performance metrics. These features include
"player_position_1", "player_position_2", "player_height", "player_weight", and
various other "player_" features related to their general, positional, offensive, and
defensive performance. Additionally, the dataset includes information about the team
and competition in which the player participated. The "team" column provides
information about the player's team, and the "competitionId" column provides
information about the competition in which the player participated.There is also a
"winner" column, which identifies the winning team for each game. This column
could potentially be used as a feature in the model to determine if a player's
performance is affected by their team's success. Overall, there are a total of 799
columns with most of them representing different metrics of a player which can be
used as features while modeling that can be used to build a machine learning model to
predict a player's overall rating

## Preprocessing
The data was imbalanced more than we anticipated and was not organized and
categorized properly. It was discovered that a large number of columns in the dataset
contained no information at all, as they were 100% null. These columns were dropped
from the dataset, leaving only those attributes that contained valuable data. For the
remaining null values, the mean imputation technique was used to fill the empty cells,
ensuring that the dataset was complete and ready for analysis.
Next we wanted to see the distribution of the dataset and found out that there were
multiple columns which depicted skewness and multimodality through density plot
distribution of multiple same prefixed variables.
Since we figured out that there were multiple columns depicting skewed and
multimodal behavior we treated them with quartile transformation and label encoding.
We have applied Quantile Transformer on selected few columns which mainly come
under the group of 'Derived' and 'Ratio' Columns to make their distribution normal
which would be good for modeling. Quantile Transformer applies a non-linear
transformation to the data that maps it to a uniform distribution, and then applies the inverse cumulative distribution function (CDF) of a normal distribution to map the
data to a Gaussian distribution. This process can help to reduce the impact of outliers and skewness in the data, and improve the performance.
For multimodal kind of distribution, we decided to convert them to object type and
then Label Encode them. The columns that were originally of type 'int' were converted
to type 'object', making them more compatible with the XGBoost regressor model that
was later used for analysis. 


![Screenshot (1143)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/6d17c575-2bc4-44ea-ac4a-88720f832369)


## Feature Importance
We considered 3 models for feature importance.
● XGBOOST
● LightGBM
● CATBOOST

### Extreme Gradient Boosting
XGBoost (Extreme Gradient Boosting) is an ensemble machine learning
algorithm that uses a gradient boosting framework to train decision trees in a parallel and optimized manner. It works by iteratively adding decision trees to the model, with each tree trained to correct the errors of the previous trees. The data was fit to the model using 100 n_estimators. To evaluate the performance of the model, the R2 score was used. The model achieved a score of 0.83 on the training data and 0.29 on the testing data

![Screenshot (1144)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/3d354377-db1d-4cec-8816-22922d9c24c9)


### Light Gradient Boosting Machine
LightGBM (Light Gradient Boosting Machine) is another gradient boosting
framework that is designed to be fast and efficient. It uses a histogram-based approach to split data into bins, which can significantly reduce the memory usage and speed up the training process. For LightGBM we used 170 n_estimators. To evaluate the
performance of the model, the R2 score was used. The model achieved a score of 0.68
on the training data and 0.32 on the testing data.

![Screenshot (1145)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/2fbbcec2-a36f-4fd3-adb0-46d87a554e4e)

CatBoost (“Category” and “Boosting”)
CatBoost is a gradient boosting framework that is designed to handle
categorical features. It uses a combination of ordered boosting and random
permutations to handle categorical data in an efficient and accurate way. The model
was implemented using catboost regressor and R2 score was used as performance
measure.The model achieved a score of 0.73 on the training data and 0.37 on the
testing data

![Screenshot (1146)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/6db11e27-a85c-4bb8-a471-100989185af6)


## Neural Network for predicting Player rating

### Data Preprocessing for neural network:
The scouting dataset contained both continuous and categorical features, and it
was necessary to preprocess the data to prepare it for use with an ANN even after the
initial preprocessing. First, the categorical features were identified and mapped to
numerical values. The winner feature was mapped to 0 for loser, 1 for winner, and 2
for draw. Similarly, the team feature was mapped to 0 for team1 and 1 for team2.
Next, the continuous and object features were concatenated and normalized
using min-max scaling to ensure that the input values were in the range [0,1]. The
resulting tensor was used as the input to the neural network model.


### Model Architecture and training:
The input layer of the neural network had 30 nodes, which corresponds to the number
of input features in the dataset. The hidden layers had 800 nodes each, and the output layer had a single node, which was the predicted player rating. The rectified linear unit (ReLU) activation function was used for all the hidden layers to introduce non-linearity in the model. The final output layer did not have any activation function. The neural network model was implemented using PyTorch, with the input size (number of features) specified by in_features and the output size (predicted player rating) specified by out_features = 1. The model was initialized using a seed value of 32 to ensure reproducibility of results.
The model was trained for a total of 800 epochs using mean squared error (MSE) loss
as the loss function. The optimizer used for training was the Adam optimizer with a
learning rate of 0.0001.
During each epoch, the gradients from the previous iteration were cleared using
optimizer.zero_grad(), and the forward pass was computed by passing the normalized
input data (train_model_input_normalized) through the neural network model. The
loss was then calculated as the MSE between the predicted output and the actual
output (train_y_out). The loss.backward() function performed the backward pass to
compute the gradients of the loss with respect to the parameters of the neural network, and optimizer.step() updated the weights of the network based on these gradients.
The model was trained on 70% of the preprocessed data, and the remaining 30% was
used for testing. The training process was monitored by observing the loss after every 10 epochs. Finally, the duration of the training process was also noted.

## Results:

### Results for the 800 feature dataset: 
After training the model, it was evaluated on a separate test set to check its accuracy. The root mean squared error (RMSE) was calculated, which is a common metric for regression problems. The model achieved an RMSE of 1.76142263, which was not what we anticipated but with highly imbalanced data we couldn’t agree more on such a result.

![Screenshot (1147)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/1b0d10b8-c5e7-4a5c-90d0-2c076b86a317)


To further analyze the model's performance, a scatter plot was created to visualize the predicted and actual values of the test set. The scatter plot showed that the predicted values were highly correlated with the actual values.

![Screenshot (1148)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/518549f1-7f03-4991-966c-8c89b2d72980)

![Screenshot (1149)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/7f88b644-6946-4b25-8c74-47404922d368)


Additionally, a density plot was created to compare the distributions of the predicted and actual values. The density plot showed that the distributions were not very accurate due to large imbalance in the dataset despite normalization.

### Results for 50 feature dataset: 
For this model, the root mean squared error (RMSE) was calculated, which is a common metric for regression problems. The model achieved an RMSE of RMSE:1.090376, which turned out to be better than our previous dataset implementation.



Similarly for the model's performance, a scatter plot was created to visualize the
predicted and actual values of the test set. The scatter plot showed that the predicted values were highly correlated with the actual values.



Finally, a density plot was created to compare the distributions of the predicted and
actual values. The density plot showed that the distributions were similar, indicating that the model's predictions were well calibrated.



Overall, the model's architecture and hyperparameters were chosen through
experimentation to achieve the best performance, and the training process was
successful in minimizing the loss function. The model's predictions were reasonably
accurate, as indicated by the low RMSE value and the scatter and density plots

