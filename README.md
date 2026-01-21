# Wine Price Analysis by Country

A comprehensive data analysis project examining the relationship between wine prices, quality ratings (points), and country of origin using a dataset of 130,000+ wine reviews.

## 📊 Project Overview

This project analyzes wine review data to understand pricing patterns across different wine-producing countries. The analysis determines which countries produce the most expensive wines, which offer the best value, and provides insights into the competitive positioning of major wine-producing regions including Italy, France, South Africa, and others.

## 🎯 Objectives

- Identify which country produces the most expensive wines on average
- Identify which country produces the most affordable wines
- Compare Italy vs South Africa vs other major wine-producing countries
- Determine whether South Africa ranks among expensive or best-value wine producers
- Analyze price distributions per country
- Provide clear rankings and actionable insights

## 📁 Dataset

- **Source**: WineMag Review Dataset (130k+ records)
- **Format**: CSV and JSON formats
- **Key Columns**:
  - `country`: Wine origin country
  - `price`: Price in USD
  - `points`: Wine quality rating (0-100 scale)
  - `province`: Wine region/province
  - `region_1`, `region_2`: Specific wine regions
  - `variety`: Grape variety
  - `winery`: Winery name

## 🔧 Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static visualizations
- **seaborn**: Enhanced statistical visualizations

## 📋 Project Structure

```
Wine/
├── wine_analysis.py              # Main analysis script
├── winemag-data-130k-v2.csv      # Dataset (CSV format)
├── winemag-data-130k-v2.json     # Dataset (JSON format)
├── plots/                        # Generated visualizations
│   ├── boxplot_price_by_country_top10.png
│   ├── bar_mean_price_by_country.png
│   └── scatter_price_vs_points_by_country.png
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Installation & Setup

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

### Running the Analysis

```bash
python wine_analysis.py
```

The script will:
1. Load and validate both CSV and JSON datasets
2. Clean data (remove missing prices/countries, filter invalid entries)
3. Compute country-level statistics (mean, median, std deviation, count)
4. Calculate best-value scores (points per price ratio)
5. Generate visualizations
6. Print comprehensive rankings and conclusions

## 📈 Key Features

### Data Processing
- **Robust data loading**: Handles both CSV and JSON formats with validation
- **Data cleaning**: Removes null prices, zero prices, and missing country data
- **Sample size filtering**: Excludes countries with fewer than 50 wines for statistical reliability

### Analysis Metrics
- **Mean price per country**: Average wine price
- **Median price per country**: Middle value (less affected by outliers)
- **Price standard deviation**: Variability in pricing
- **Value score**: Points-to-price ratio (best-value metric)

### Visualizations
1. **Boxplot**: Price distribution for top 10 countries by wine count (log scale)
2. **Bar Chart**: Average price comparison across all countries
3. **Scatter Plot**: Price vs. quality rating, colored by country (top 6 countries)

## 📊 Key Findings

### Most Expensive Wine Countries
1. **England** - $51.68 average price
2. **Germany** - $42.26 average price
3. **France** - $41.14 average price
4. **Hungary** - $40.65 average price
5. **Italy** - $39.66 average price

### Most Affordable Wine Countries (with ≥50 wines)
1. **Bulgaria** - $14.65 average price
2. **Romania** - $15.24 average price
3. **Moldova** - $16.75 average price
4. **Georgia** - $19.32 average price
5. **Chile** - $20.79 average price

### Best Value Countries (Highest Points per Dollar)
1. **Romania** - Best value score
2. **Bulgaria**
3. **Moldova**
4. **Chile**
5. **Portugal**
6. **Argentina**
7. **South Africa** - Ranked 8th overall, 3rd among major producers

### Focus Country Comparison

| Country | Mean Price (USD) | Median Price (USD) | Wine Count | Value Rank |
|---------|-----------------|-------------------|------------|------------|
| France | $41.14 | $25.00 | 17,776 | 6th |
| Italy | $39.66 | $28.00 | 16,914 | 22nd |
| United States | $36.57 | $30.00 | 54,265 | 26th |
| Spain | $28.22 | $18.00 | 6,573 | 4th |
| South Africa | $24.67 | $19.00 | 1,293 | 3rd |
| Argentina | $24.51 | $17.00 | 3,756 | 2nd |
| Chile | $20.79 | $15.00 | 4,416 | 1st |

### Insights & Conclusions

- **South Africa** is relatively **affordable** (ranked 18th out of 26 countries by price) but offers **excellent value** (3rd best among major wine-producing countries)
- **Italy** is relatively **expensive** (5th highest average price) but has lower value-for-money ranking
- **Chile** and **Argentina** dominate the best-value category among major producers
- **France** and **Italy** maintain premium pricing positioning
- **Romania**, **Bulgaria**, and **Moldova** offer exceptional value but have smaller market representation

## 🎓 Skills Demonstrated

- **Data Wrangling**: Handling large datasets (130k+ records) with multiple formats
- **Data Cleaning**: Missing value treatment, outlier handling, data validation
- **Statistical Analysis**: Mean, median, standard deviation, aggregation by groups
- **Data Visualization**: Multiple chart types (boxplots, bar charts, scatter plots) with proper scaling
- **Code Organization**: Modular functions, clear structure, type hints, comprehensive documentation
- **Business Insights**: Converting data analysis into actionable conclusions

## 📝 Code Quality

- Clean, well-structured Python code with type hints
- Comprehensive function documentation
- Separation of concerns (data loading, cleaning, analysis, visualization, reporting)
- Configurable parameters (minimum sample size, file paths)
- Error handling and data validation

## 📧 Contact

For questions or collaborations, please reach out through GitHub.

---

**Note**: This analysis assumes all prices are in USD and uses standard statistical methods for comparison. Results are based on the WineMag dataset and may not represent all wines in each country.

