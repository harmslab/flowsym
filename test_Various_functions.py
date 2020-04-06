# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:29:28 2020

@author: Michael
"""

#from Various_functions import square_positive_int
import pytest
from Various_functions import create_sample
import pandas as pd
# Can make a class of multiple tests
# IMPORTANT: Name your class as TestXXX


def test_create_sample():
    """
    test the create sample function to make sure we made our cell dataframe correctly
    """
    test = create_sample(size=100,ex=[300,400,500],em=[350,450,550]) # Params for test function
    
    assert len(test) == 100   # Is dataframe correct size from input step?
    assert type(test) == type(pd.DataFrame())   # Did we actually output a dataframe?
    assert list(test.columns) == ['Excitation','Emission']  # Are these the two columns?
    
    for index, value in enumerate(test['Excitation']):   # Do emission wavelengths math excitation?
        assert test['Emission'][index] == value+50  



# Now that we've tested the create sample function, make a fixture for following function tests
@pytest.fixture()
def data():
    dataframe = create_sample(size=100,ex=[300,400,500],em=[350,450,550])
    return dataframe

# example
def test_column_nums(data):
    assert len(list(data.columns)) == 2
