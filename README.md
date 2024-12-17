# World Happiness Report Analysis Dashboard 😊

An interactive Streamlit dashboard that visualizes and analyzes the World Happiness Report data from 2015-2019. This application provides comprehensive insights into global happiness trends, correlations between different factors, and country clustering based on various metrics.

## Features

### 1. Overview
- Display of recent data samples and statistical summaries
- Interactive choropleth map showing global happiness score distribution
- Quick insights into happiness metrics across countries

### 2. Correlation Analysis
- Interactive correlation matrix visualization
- Comparative analysis of different happiness metrics
- Top 10 country rankings by various factors (GDP, Social Support, etc.)

### 3. Time Series Analysis
- Track happiness score trends from 2015 to 2019
- Compare multiple countries simultaneously
- Interactive line charts with country selection

### 4. Clustering Analysis
- K-means clustering based on social support and health expectancy
- Adjustable number of clusters (2-5)
- Detailed cluster statistics and country groupings
- Visual representation of clusters with highlighted key countries

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/world-happiness-analysis.git
cd world-happiness-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Required Dependencies
- streamlit
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- plotly

## Dataset Requirements
Place the following CSV files in the project directory:
- 2015.csv
- 2016.csv
- 2017.csv
- 2018.csv
- 2019.csv

You can download these files from the [World Happiness Report dataset on Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness/data).

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser. Use the sidebar to navigate between different types of analysis:
- Overview
- Correlation Analysis
- Time Series Analysis
- Clustering Analysis

## Features in Detail

### Overview
- Displays a sample of the 2019 dataset
- Shows statistical summary of the data
- Interactive choropleth map of global happiness scores

### Correlation Analysis
- Interactive correlation matrix of all numeric variables
- Top 10 countries by different metrics
- Customizable metric selection

### Time Series Analysis
- Compare happiness trends across multiple countries
- Interactive country selection
- Year-over-year trend visualization (2015-2019)

### Clustering Analysis
- K-means clustering visualization
- Adjustable number of clusters
- Detailed cluster statistics
- Highlighted key countries in each cluster

## Data Source
The data used in this project comes from the World Happiness Report dataset available on Kaggle. The dataset includes various factors that contribute to national happiness scores, such as:
- GDP per capita
- Social support
- Healthy life expectancy
- Freedom to make life choices
- And more

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
