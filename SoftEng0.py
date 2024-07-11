import csv
import pandas as pd
import seaborn as sns


heartds = pd.read_csv('https://raw.githubusercontent.com/orionmc/SoftEng0/main/heart0.csv')

#For quick overview of the data we can use either tail or head functions
#print(heartds.tail(6))
#print(heartds.head(6))
#heartds.to_csv(r'c:\mytestprj\heart0.csv', index=False) # write the dataframe to csv file on specified location.

# check for missing values in the dataset
#print(heartds[heartds.isna().any(axis=1)]) # print the rows with missing values
print(heartds.isna())   # check for missing values in the dataset - print False if no missing values and True if missing values are present

# rectify the missing values in the dataset can be done in multiple ways
# fillna() with parameter ffill or bfill can be used to fill the missing values with the previous or next values
# they can be filled with 0-s or with median values 
# or we can simply use dropna() to drop the rows with missing values

heartds_no_missing = heartds.dropna()   # drop the rows with missing values

heartds[heartds.duplicated()]           # check for duplicate rows in the dataset
heartds.drop_duplicates(inplace=True)   # drop the duplicate rows in the dataset





