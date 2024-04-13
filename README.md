
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


# Project Title

A brief description of what this project does - Problem Statement you're trying to solve!


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results
![alt text](https://github.com/123ofai/Demo-Project-Repo/blob/main/results/test.png)

As you can see from the above image, the model has signifcant amount of error in <x, y, z regions>

| Metric        | Value         |
| ------------- | ------------- |
| R2 Score      | 0.11          |
| MSE           | 0.76          |
| sdfsdfs       | 0.77          | 

The above quant results show that <>
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

