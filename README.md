
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


# Project Title

Highway Traffic Forecasting: ML-Powered Traffic Volume Prediction


## Implementation Details

- Dataset: Metro Interstate Traffic Volume Dataset (view below for more details)
- Model: [HistGradientBoostingRegressor]('https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html')
- Input: 8 features - Holiday, Temp, Weather Description ...
- Output: Traffic Volume

## Dataset Details

[This dataset was obtained from this repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
)

Metro Interstate Traffic Volume  dataset is a collection of traffic volume data observed on a section of interstate highway in the Minneapolis-St Paul metropolitan area in Minnesota, USA. This dataset includes hourly traffic volume measurements along with corresponding attributes such as date, time, weather conditions, and holiday indicators. The data spans from 2012 to 2018, providing a comprehensive view of traffic patterns over several years. This dataset is valuable for studying and predicting traffic volume fluctuations based on various factors, making it suitable for machine learning tasks such as regression and time series analysis.

### Varibles Table of the above dataset
 ![alt text](https://github.com/vasanthgx/traffic_volume/blob/main/images/dataset1.png)

### Additional Variable Information
 ![alt text](https://github.com/vasanthgx/traffic_volume/blob/main/images/dataset2.png)


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

