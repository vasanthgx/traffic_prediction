

![logo](https://github.com/vasanthgx/regularisation_in_ml/blob/main/images/resizedlogo1.png)


# Project Title


Highway Traffic Forecasting: ML-Powered Traffic Volume Prediction
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>


## Implementation Details

- Dataset: Metro Interstate Traffic Volume Dataset (view below for more details)
- Model: [HistGradientBoostingRegressor]('https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html')
- Input: 8 features - Holiday, Temp, Weather Description ...
- Output: Traffic Volume.

## Dataset Details

[This dataset was obtained from this repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

Metro Interstate Traffic Volume  dataset is a collection of traffic volume data observed on a section of interstate highway in the Minneapolis-St Paul metropolitan area in Minnesota, USA. This dataset includes hourly traffic volume measurements along with corresponding attributes such as date, time, weather conditions, and holiday indicators. The data spans from 2012 to 2018, providing a comprehensive view of traffic patterns over several years. This dataset is valuable for studying and predicting traffic volume fluctuations based on various factors, making it suitable for machine learning tasks such as regression and time series analysis.

### Varibles Table of the above dataset

 ![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/dataset1.png)

### Additional Variable Information

 ![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/dataset2.png)


## Evaluation and Results

### Exploring the dataset statsitics

- We find from the evaluation of the dataset that all the features have null values.
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/dataset0.png)

- We have poor representation of the features 'rain_1h' and 'snow_1h' in the Dataset.
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/rain_snow.png)

- We have 3 category features 'holiday','weather_main', 'weather_description'.
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/category_feature.png)

**Inference From the above Dataset**
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/dataset_inference.png)

### Visualization - Univariate analysis

- Let us further explore the data distribution of each of the 4 numerical features plus the target of the data 'traffic_volume'.
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/trafficVol_temp_univariate.png)
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/rain_snow_graph.png)
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/clouds_all_univariate.png)
**We can clearly see from the above graphs, there is poor representation of rain_1h and snow_1h in the dataset**

- Next we will explore the Categorical features 'holiday', and 'weather_main'.


![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/holiday_graph.png)
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/trafficVol_temp_univariate.png)

### Visualization - Bivariate analysis

- Clouds All feature vs Traffic Volume
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/clouds_all_vs_traffic_volume_graph.png)

- Weather Main feature vs Traffic Volume
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/weather_main_vs_traffic_volume_graph.png)



### Correlation between the features
  Correlation tests are often used in feature selection, where the goal is to identify the most relevant features (variables) for a predictive model. Features with high correlation with the target variable are often considered important for prediction. However, it's essential to note that correlation does not imply causation, and other factors such as domain knowledge and data quality should also be considered in feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_initial_dataset.png)

## Data Cleaning and Pre Processing
- Pre-processing (Cleaning): Address missing (NULL) values - drop or imputation.
    - we will use the ffill() method
    ```
    data.ffill(inplace = True)
    ```
	
- Since we have already seen poor reperesentation of 'snow_1h' and 'rain_1h', and similarity between weather_main and  'weather_description' we will drop the three features for the model.
    ```
    data1 = data.drop(['snow_1h', 'rain_1h','weather_description'] , axis =1)
    ```
- Converting 'holiday' feature into just holiday and 'unknown'.
    ```
    data1['holiday'] = data1['holiday'].apply(lambda x: 'unknown' if pd.isna(x) else 'holiday' ) 
    ```
- Next we will first convert the 'date_time' feature into a pandas datetime object.
    ```
    data1['date_time'] = pd.to_datetime(data1['date_time'], format = '%d-%m-%Y %H:%M')
    ```
- We now extract the 'year', 'month', 'weekday' and 'hour' from the datetime object.
    ```
    data1['year'] = data1['date_time'].dt.year
    data1['month'] = data1['date_time'].dt.month
    data1['weekday'] = data1['date_time'].dt.weekday
    data1['hour'] = data1['date_time'].dt.hour
    ```
- Next we will now divide the 24 hours of the day into before_sunrise, after_sunrise, afternoon and night categories.
    ```
    data1['hour'].unique()
    ```
- We will create a function ,which will split the hours into the 4 categories.
    ```
    def day_category(hour):
        day_section = ''
        if hour in [1,2,3,4,5]:
            day_section = 'before_sunrise'
        elif hour in [6,7,8,9,10,11,12]:
            day_section = 'after_sunrise'
        elif hour in [13,14, 15, 16, 17, 18]:
            day_section = 'evening'
        else :
            day_section = 'night'
        return day_section
    ```
- Using the map() function to loop through the 'hour' feature and based on the hour - value we will allot the 4 day-sections. This way we will create one more feature 'day_section' in our existing dataset.
    ```
    data1['day_section'] = data1['hour'].map(day_category)
    ```
- Next we use the pd.get_dummies function to do one hot encoding of the categorical features 'holiday', 'weather_main' and day section.
    ```
    data1 = pd.get_dummies(data1, columns =['holiday', 'weather_main','day_section'])
    ```
- Finally we set the feature 'date_time' as row index in our dataset.
    ```
    data1.set_index('date_time',inplace = True)
    ```
### Correlation testing - second time
- After the above feature engineering. 

    ```
    corr_data1 = data1.corr()
    fig, ax = plt.subplots(figsize = (15, 10))
    plt.xticks(rotation =45)
    sns.heatmap(corr_data1, annot = True, linewidths = .5, fmt = '.1f', ax = ax)
    plt.show()
    ```
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_after_feature_engineering.png)

## Feature Importance and Selection Using Random Forest Regressor

Feature importance and selection with the Random Forest Regressor involve identifying the most influential features in predicting the target variable.

**Feature Importance:** Random Forest Regressor calculates feature importance based on how much the tree nodes that use that feature reduce impurity across all trees in the forest. Features that lead to large reductions in impurity when used at the root of a decision tree are considered more important. Random Forest assigns a score to each feature, indicating its importance. Higher scores signify more important features.

**Visualizing Feature Importance:** Plotting the feature importance scores can provide insights into which features are most relevant for prediction. This visualization can aid in understanding the data and making decisions about feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/feature_selection.png)

In summary, feature importance and selection with Random Forest Regressor involve identifying and prioritizing features based on their contribution to predicting the target variable. This process can enhance model performance, interpretability, and understanding of the underlying data.

## Model Development

- we will select the top 7 features that we got from the Random Forest Regressor
    ```
    important_features = [ 'hour','temp','weekday','day_section_night','month', 'year','clouds_all']
    ```
- Splitting the dataset into training and **validation set**. This validation set is to test our model internally before submitting it to the test set
    - *Note : we have already been provided the test data set for the hackathon*

- Scaling : we do the scaling of the data using the StandardScaler() function from sklearn

- Experimenting with different models

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/experimenting_models.png)

- Selecting the best model
    ```
    regrh = HistGradientBoostingRegressor(random_state=32)
    regrh.fit(x_train_scaled, y_train)
    y_pred = regrh.predict(x_test_scaled)
    print(f"r2 score : {r2_score(y_test, y_pred)} \n mean squared error : {mean_squared_error(y_test, y_pred)} \n mean absolute error : {mean_absolute_error(y_test,y_pred)} ")
    ```

## Testing and Creating Output CSV

- we repeat the same process of data cleaing, pre processing, scaling etc with the test data.
- finally we submit the submission file.























## Key Takeaways

What did you learn while building this project? What challenges did you face and how did you overcome them?


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

What are the future modification you plan on making to this project?

- Try more models

- Wrapped Based Feature Selection


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the linear regression model work?

Answer 1

#### How do you train the model on a new dataset?

Answer 2

#### What is the California Housing Dataset?

Answer 2
## Acknowledgements

All the links, blogs, videos, papers you referred to/took inspiration from for building this project. 

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at fake@fake.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

