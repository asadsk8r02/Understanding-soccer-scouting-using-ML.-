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


![Screenshot (1143)](https://github.com/asadsk8r02/Understanding-soccer-scouting-using-ML.-/assets/53692166/22beff64-9f1c-4d6b-9d20-559204f5ef55)



