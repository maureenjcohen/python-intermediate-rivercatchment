"""Module containing models representing catchment data.

The Model layer is responsible for the 'business logic' part of the software.

Catchment data is held in a Pandas dataframe (2D array) where each column contains
data for a single measurement site, and each row represents a single measurement
time across all sites.
"""

import pandas as pd
import numpy as np
from functools import reduce

def read_variable_from_csv(filename):
    """Reads a named variable from a CSV file, and returns a
    pandas dataframe containing that variable. The CSV file must contain
    a column of dates, a column of site ID's, and (one or more) columns
    of data - only one of which will be read.

    :param filename: Filename of CSV to load
    :return: 2D array of given variable. Index will be dates,
             Columns will be the individual sites
    """
    dataset = pd.read_csv(filename, usecols=['Date', 'Site', 'Rainfall (mm)'])

    dataset = dataset.rename({'Date':'OldDate'}, axis='columns')
    dataset['Date'] = [pd.to_datetime(x,dayfirst=True) for x in dataset['OldDate']]
    dataset = dataset.drop('OldDate', axis='columns')

    newdataset = pd.DataFrame(index=dataset['Date'].unique())

    for site in dataset['Site'].unique():
        newdataset[site] = dataset[dataset['Site'] == site].set_index('Date')["Rainfall (mm)"]

    newdataset = newdataset.sort_index()

    return newdataset

def daily_total(data):
    """Calculate the daily total of a 2D data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).sum()

def daily_mean(data):
    """Calculate the daily mean of a 2D data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).mean()


def daily_max(data):
    """Calculate the daily max of a 2D data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).max()


def daily_min(data):
    """Calculate the daily min of a 2D data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).min()

def data_normalise(data):
    """ Normalise data between 0 and 1 column-wise."""
    col_max = np.array(np.max(data, axis=0))
    return data / col_max[np.newaxis, :]

def daily_above_threshold(site_id, data, threshold):
    """Determine whether or not each data value exceeds a given threshold for a given site.

    :param site_id: The identifier for the site column
    :param data: A 2D Pandas data frame with measurement data. Columns are measurement sites.
    :param threshold: A threshold value to check against
    :returns: A boolean list representing whether or not each data point for a given site exceeded the threshold
    """
    return list(map(lambda x: x > threshold, data[site_id]))

def data_above_threshold(site_id, data, threshold):
    """ """

    def count_above_threshold(a, b):
        print(a,b)
        if b: 
            return a + 1
        else:
            return a

    above_threshold = map(lambda x: x > threshold, data[site_id])
    return reduce(count_above_threshold, above_threshold, 0)

class MeasurementSeries:
    def __init__(self, series, name, units):
        self.series = series
        self.name = name
        self.units = units
        self.series.name = self.name
    
    def add_measurement(self, data):
        self.series = pd.concat([self.series,data])
        self.series.name = self.name
    
    def __str__(self):
        if self.units:
            return f"{self.name} ({self.units})"
        else:
            return self.name


class Site:
    def __init__(self,name):
        self.name = name
        self.measurements = {}
    
    def add_measurement(self, measurement_id, data, units=None):    
        if measurement_id in self.measurements.keys():
            self.measurements[measurement_id].add_measurement(data)
        
        else:
            self.measurements[measurement_id] = MeasurementSeries(data, measurement_id, units)
    
    @property
    def last_measurements(self):
        return pd.concat(
            [self.measurements[key].series[-1:] for key in self.measurements.keys()],
            axis=1).sort_index()
    
    def __str__(self):
        return self.name