#!/usr/bin/env

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lightweight adjustable framework for simple exploratory data analysis and visualisation 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Importing the necessary modules
import pandas as pd
import sklearn as skl
import numpy as np
import pprint as pp
import os, sys, shutil, pathlib
import time
#import sqlite3
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pathlib
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sets the working directory -- or, workspace
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("The current working directory is: ", os.getcwd(), '\n')
# Enter the path without quotations or // separation
path = input("Please enter the path to the working directory: ")
os.chdir(path)
file_location = input("Please enter the file location: ")
file_location = str(file_location)


# FIGURE OUT WHICH CONNECTION METHOD TO USE BASED ON FILE EXTENSION
file_name, file_extension = os.path.splitext(file_location)
if file_extension == '.csv':
    data = pd.read_csv(file_location, encoding = 'iso-8859-1')
elif file_extension == '.xls' or '.xlsx':
    data = pd.read_excel(file_location)

#tree = ET.parse(file_location)
#root = tree.getroot()

#con = sqlite3.connect(file_location)
#data = pd.read_sql_query("SELECT * FROM Data;", con)
#con.close()

print('~~~ File Loaded !! ~~~', '\n')




# -----------------------------------------------------------------------------------------------------------
# Minor processing steps
# -----------------------------------------------------------------------------------------------------------

import pprint as pp

print(data.head(), '\n')
print(data.describe(), '\n')
print(data.shape, '\n')
        
df_length = data.shape[0]
nulls = data.isnull().sum()
variables = pd.Series(data=data.columns.values, index=None)
print(variables, '\n')
        
nulls = nulls.to_dict()
to_drop = []
# If a column has greater than 20% of total null values - it gets dropped
for k, v in nulls.items():
    if (v == df_length):
        to_drop.append(k)
data.drop(to_drop, axis=1, inplace=True)
data.drop_duplicates(keep='first', inplace=True)

print("AFTER DROPPING EGREGIOUSLY NULL COLUMNS, AND HANDLING DUPLICATE OBSERVATIONS, THE SHAPE OF THE DATAFRAME IS:")   
print(data.shape, '\n')
  




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SUBSETTING THE MAIN DATAFRAME BY VARIABLES' DATA TYPE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# to be tested
# Initializing positional arguments for the dataframe
shape = data.shape
obs_num = int(shape[0])
var_num = int(shape[1])
print(data.dtypes, '\n')
                   
    
# Creating dtype subsets
numerics = data.select_dtypes(include=['float64']).copy()
integers = data.select_dtypes(include=['int64']).copy()
dates = data.select_dtypes(include=['datetime64[ns]']).copy()
objects = data.select_dtypes(include=['object']).copy()
booleans = data.select_dtypes(include=['bool']).copy()



if len(numerics.columns.values) > 0:
    numerics.fillna(value=0.0, inplace=True)
elif len(numerics.columns.values) == 0:
    exit

# POSSIBLE FIX TO INT BASE 10 ValueError
#for column in integers.columns:
#        for elem in column:
#            elem = int(float(elem)); 


if len(integers.columns.values) > 0:
    integers.fillna(value=0, inplace=True)
elif len(integers.columns.values) == 0:
    exit

# COMBINING NUMERIC(FLOAT) AND INTEGER VARIABLES INTO A COMBINED 'NUMBER' DF
numbers = pd.concat([numerics, integers], axis=1, sort=False)   

for column in objects.columns:
    print(objects[column].value_counts(), '\n')

for column in numerics.columns:
    print(numerics[column].describe(), '\n')


for obj in objects.columns:
    colName = str(obj) + 'code'
    objects[obj] = objects[obj].astype('category')
    objects[colName] = objects[obj].cat.codes

# Calculating the quantiles of a numeric variable/dataset
#for column in numerics.columns:
#    numerics['quantile'] = numerics[column].quantile([0.25,0.5,0.75])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# THIS CODE CAN BE CLEANED UP

print(variables, '\n')

print('DO YOU NEED TO FILTER BY SPECIFIC VARIABLES?')
confirmation = input('ENTER y or n: ')
filters = input('ENTER THE VARIABLES AS A COMMA SEPARATED LIST: ')
filter_vars = [x.strip() for x in filters.split(',')]

additional_var_num = len(filter_vars)
additional_var_num = int(additional_var_num)

# SELECTING THE ARITHMETIC GROUPING METHOD
grouping_method = input('HOW DO YOU WANT TO CONSOLIDATE THE DATA - SUM OR AVERAGE? ')



if (confirmation == 'y') and (grouping_method == 'sum' or 'SUM' or 'Sum'):
    # CREATING ALTERNATE DATAFRAMES AROUND EACH OF THE SPECIFIED COLUMNS
    for var in filter_vars:
        var = str(var)
        data_b = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).sum())
        filter_vars.remove(var)
    for var in filter_vars:
        length = len(filter_vars)
        data_c = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).sum())
        filter_vars.remove(var)
        if len(filter_vars) > 0:
            for var in filter_vars:
                data_d = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).sum())
                filter_vars.remove(var)
            for var in filter_vars:
                data_e = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).sum())
                filter_vars.remove(var)
        elif len(filter_vars) == 0:
            break
    if grouping_method == 'average' or 'AVERAGE' or 'Average':
        for var in filter_vars:
            var = str(var)
            data_b = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).mean())
            filter_vars.remove(var)
        for var in filter_vars:
            length = len(filter_vars)
            data_c = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).mean())
            filter_vars.remove(var)
            if len(filter_vars) > 0:
                for var in filter_vars:
                    data_d = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).mean())
                    filter_vars.remove(var)
                for var in filter_vars:
                    data_e = pd.DataFrame(data=data.groupby(by=[filter_vars[0]], as_index=False).mean())
                    filter_vars.remove(var)
            elif len(filter_vars) == 0:
                break
                
elif confirmation == 'n':
    exit()
    

if additional_var_num == 1:
    print(data_b.head(), '\n')
elif additional_var_num <= 2:
    print(data_b.head(), '\n')
    print(data_c.head(), '\n')
elif additional_var_num <= 3:
    print(data_b.head(), '\n')
    print(data_c.head(), '\n')
    print(data_d.head(), '\n')
elif additional_var_num <= 4:
    print(data_b.head(), '\n')
    print(data_c.head(), '\n')
    print(data_d.head(), '\n')
    print(data_e.head(), '\n')



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Extracting columns that contain unique identifiers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

#identifiers = list(variables[variables.str.contains('ID') == True])
#identifiers_o = list(variables[variables.str.contains('id') == True])
#identifiers_t = list(variables[variables.str.contains('Id') == True])
#identifiers_th = list(variables[variables.str.contains('i.d.') == True])
#identifiers_f = list(variables[variables.str.contains('I.D.') == True])
#client_naming = list(variables[variables.str.contains('name') == True])
#client_naming_alt = list(variables[variables.str.contains('client') == True])
#client_naming_altO = list(variables[variables.str.contains('Client') == True])
#client_naming_altT = list(variables[variables.str.contains('CLIENT') == True])
#fund_naming = list(variables[variables.str.contains('fund') == True])
#fund_naming_alt = list(variables[variables.str.contains('Fund') == True])
#fund_naming_altO = list(variables[variables.str.contains('FUND') == True])

#mapping_cols = [identifiers, identifiers_o, identifiers_t, identifiers_th, identifiers_f, 
#                client_naming, client_naming_alt, client_naming_altO, client_naming_altT,]

#mapping_fields = []

#for col in mapping_cols:
#    if len(col) > 0:
#        mapping_fields.append(col); 

#identifiers = []

#for identifier in mapping_fields:
#    identifiers.append(identifier);

#vector_list = []

#for identifier in identifiers:
#    series = data[identifier]
#    vector_list.append(series)

#identifying_data = pd.concat(objs=vector_list, axis=1)
#print(identifying_data.head(), '\n')
        

#historical_mapping = pd.read_excel('H:\\Application_Data\\MAIN\\mapping\\identifier_archive\\mapping.xlsx', header=0)

#mapping_zip = pd.merge(left=historical_mapping, right=identifying_data, left_on='Client Code', right_on=


# Writing the dataframe to a network drive for storing and appending

#with pd.ExcelWriter('H:\\Application_Data\\MAIN\\mapping\\identifier_archive\\mapping.xlsx') as writer:  # doctest: +SKIP
#     mapping_zip.to_excel(writer, sheet_name='Sheet_name_1', index=False)
     


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining a function that will take two series as inputs and plot them against eachother
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Example histogram and pdf using an actual calculated mean and standard deviation/variance

def gauss_dist(x):
    # The input is/should be a series
    # Set the style 
    sns.set(style='darkgrid')

    # Define the measures

    average = x.mean()
    standardDev = x.std()

    s = np.random.normal(average, standardDev, 1000)
    bins = 30
    plt.hist(s, 30)
    plt.plot(bins, 1/(standardDev * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - average)**2 / (2 * standardDev**2)),
             linewidth=2, color='r')

    plt.title(str(x.name))

    plt.show()

    
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# Calling the normal distribution function on the subset of integer/numeric variables
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#if len(numerics.columns.values) > 0:
#    for column in numerics:
#        viz_data = pd.Series(data=numerics[column], index=None)
#        if viz_data.isnull().sum() == len(viz_data):
#            exit
#        elif viz_data.isnull().sum() != len(viz_data):
#            gen_distribution(viz_data);
#        elif len(numerics.columns.values) == 0:
#            exit


#if len(integers.columns.values) > 0:
#    for column in integers:
#        viz_data = pd.Series(data=integers[column], index=None)
#        if viz_data.isnull().sum() == len(viz_data):
#            exit
#        elif viz_data.isnull().sum() != len(viz_data):
#            gen_distribution(viz_data)
#        elif len(integers.columns.values) == 0:
#            exit

if len(numbers.columns.values) > 0:
    for column in numbers:
        viz_data = pd.Series(data=numbers[column], index=None)
        if viz_data.isnull().sum() == len(viz_data):
            break
        elif viz_data.isnull().sum() != len(viz_data):
            gauss_dist(viz_data);
        elif len(numbers.columns.values) == 0:
            break

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Defining a function that will take two inputs and scatter/RegPlot against eachother

def scatter_reg_plt (x, y):

    # The two function inputs should be Series of the same length
    x = pd.Series(data=x, index=None)
    y = pd.Series(data=y, index=None)


    sns.set(style='darkgrid', palette='deep')
    
    plt.scatter(x=x, y=y, marker='o', alpha=0.5)

    plt.show()
    
    sns.regplot(x=x, y=y, marker="+", data=data, ci=95)
    
    plt.show()

    #R^2 FUNCTION
    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    sns.jointplot(x=x, y=y, kind="reg", stat_func=r2)
    plt.show()
    

# Loop through column pairs and scatter against eachother



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to product a correlation matrix based on a dataframe of variables

def correlation_visual (data):
    
    corr = data.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    print(corr, '\n')

    plt.show()


correlation_visual(data)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot with +/- standard deviations

import chart_studio.plotly as py
import plotly.graph_objects as go

def deviation_plot (data):

    x_label = input('Enter the x-axis variable: ')
    y_label = input('Enter the y-axis variable: ')

    x = data[x_label]
    y = data[y_label]

    y_dev = y.std()


    upper_bound = go.Scatter(
        name='Upper Bound',
        x=x,
        y=y+y_dev,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Trend',
        x=x,
        y=y,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=x,
        y=y-y_dev,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')

    # Trace order can be important
    # with continuous error bars
    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title=str(y_label)),
        title=(str(y_label)+' '+'against'+' '+str(x_label)),
        showlegend = False)

    fig = go.Figure(data=data, layout=layout)
    #py.iplot(fig, filename='pandas-continuous-error-bars')
    fig.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Horizontal bar charts for categories and object variables

def horizontal_bar (data):
    
    x_label = input('Enter the x variable: ')
    xt_label = input('Enter a comparative x variable: ')
    y_label = input('Enter the y variable: ')

    x = data[x_label]
    xt = data[xt_label]
    y = data[y_label]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x,
        name=(str(x_label)),
        orientation='h',
        marker=dict(
            color='rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)')
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=xt,
        name=(str(xt_label)),
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)')
        )
    ))

    fig.update_layout(barmode='stack')
    fig.show()

#, width=3
