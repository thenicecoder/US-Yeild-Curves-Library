#Import all librarie
import numpy as np
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ipywidgets import SelectionSlider
from ipywidgets import interact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class YieldCurveData:
    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year
        self.df = pd.DataFrame()  # In practice, this would be filled with real data
        self.scaler = StandardScaler()
        self.df_filled = None  # Initialize df_filled
        self.df_standardized = None  # Initialize df_standardized
        self.principal_df = None  # Initialize principal_df
        self.explained_variance_ratio = None  # Initialize explained_variance_ratio


    def fetch_year_data(self, year):
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
        yield_curve_tab = pd.read_html(url)
        return yield_curve_tab[0]

    def clean_df(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.drop(['20 YR', '30 YR', 'Extrapolation Factor',
                      '8 WEEKS BANK DISCOUNT', 'COUPON EQUIVALENT', '17 WEEKS BANK DISCOUNT',
                      'COUPON EQUIVALENT.1', '52 WEEKS BANK DISCOUNT', 'COUPON EQUIVALENT.2'], axis=1)
        return df

    def concatenate_data(self):
        for year in range(self.start_year, self.end_year + 1):
            year_data = self.fetch_year_data(year)
            cleaned_data = self.clean_df(year_data)
            self.df = pd.concat([self.df, cleaned_data])

        self.df = self.df.sort_index()

    def get_data(self):
        if self.df.empty:
            self.concatenate_data()
        return self.df
    
    def plot_yield_curve_by_date(self, date_str):
        if self.df.empty:
            self.concatenate_data()

        try:
            # Convert the string to a datetime object
            date = pd.to_datetime(date_str)
            # Select the data for the specified date
            if date in self.df.index:
                row = self.df.loc[date]
                yield_values = np.array(row.values, dtype=float)  # Convert the values to a 1D numpy array, ensuring they are floats

                y_min = self.df.min().min()  # Find the minimum value in the entire DataFrame
                y_max = self.df.max().max()  # Find the maximum value in the entire DataFrame

                # Plot the yield curve for the specified date
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(yield_values)), yield_values, marker='o', linestyle='-', color='b')
                plt.xlabel('Maturity')
                plt.ylabel('Yield')
                plt.title(f'Yield Curve for {date.strftime("%Y-%m-%d")}')
                plt.ylim(y_min, y_max)  # Set the y-axis limits
                plt.grid(True)
                plt.xticks(range(len(yield_values)), row.index, rotation=45)  # Set the x-ticks to be the maturity terms
                plt.show()
            else:
                print(f"No data available for {date.strftime('%Y-%m-%d')}.")
        except Exception as e:
            print(f"Error with the provided date '{date_str}': {e}")
    
    def interactive_yield_curve_plot(self):
        if self.df.empty:
            self.concatenate_data()

        # Create a selection slider that has all the dates in the DataFrame
        date_slider = SelectionSlider(
            description='Date',
            options=self.df.index.strftime('%Y-%m-%d').tolist(),
            continuous_update=False
        )

        @interact(date=date_slider)
        def update_plot(date):
            date = pd.to_datetime(date)
            row = self.df.loc[date]
            yield_values = np.array(row.values, dtype=float)

            y_min = self.df.min().min()
            y_max = self.df.max().max()

            plt.figure(figsize=(10, 6))
            plt.plot(range(len(yield_values)), yield_values, marker='o', linestyle='-', color='b')
            plt.xlabel('Maturity')
            plt.ylabel('Yield')
            plt.title(f'Yield Curve for {date.strftime("%Y-%m-%d")}')
            plt.ylim(y_min, y_max)
            plt.grid(True)
            plt.xticks(range(len(yield_values)), row.index, rotation=45)
            plt.show()

    def fill_nan_with_row_average(self,df):
        for col in df.columns:
            # Check if the entire column is NaN
            if df[col].isna().all():
                # Fill the entire column with the row average excluding itself
                df[col] = df.apply(lambda row: row[row.index != col].mean(), axis=1)
            else:
                # Fill NaN values with the average of adjacent cells in the same row
                for idx in range(len(df)):
                    if pd.isna(df.loc[df.index[idx], col]):
                        # Get adjacent cells in the same row
                        adjacent_values = df.loc[df.index[max(0, idx-1):min(idx+1, len(df)-1)], col].dropna()
                        # If there are adjacent cells, take their average
                        if not adjacent_values.empty:
                            df.loc[df.index[idx], col] = adjacent_values.mean()
                        else:
                            # If there are no adjacent cells, take the row average excluding the current NaN column
                            df.loc[df.index[idx], col] = df.loc[df.index[idx], df.columns != col].mean()
        return df

    def perform_pca(self):
        if self.df.empty:
            self.concatenate_data()
        df_filled=self.fill_nan_with_row_average(self.df)
        df_standardized = self.scaler.fit_transform(df_filled)

        # Applying PCA
        pca = PCA(n_components=3)  # We'll look at the first three principal components
        principal_components = pca.fit_transform(df_standardized)

        # Create a DataFrame for the principal components
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['Level', 'Slope', 'Curvature'],
                                    index=df_filled.index)

        # Explained variance ratio for the principal components
        explained_variance_ratio = pca.explained_variance_ratio_
        return principal_df, explained_variance_ratio
