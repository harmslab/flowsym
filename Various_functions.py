# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:57:16 2020

@author: Michael
"""
import numpy as np
import pandas as pd
import math



def create_sample(size,ex,em):
    """
    This is a function that takes a defined dataframe length for number of samples (int)
    and excitation and emission wavelengths (list,list)
    """
    
    # Check to make sure the inputs were of correct type 
    if type(size) != int:
        raise TypeError("size cannot be of type: "+str(type(size)))
    elif type(ex) != list:
        raise TypeError("Ex cannot be of type: "+str(type(ex)))
    elif type(em) != list:
        raise TypeError("Em cannot be of type: "+str(type(em)))
        
    # Set dictionary to be made into dataframe
    data_dict = {'Excitation': [],'Emission': []}
    
    # Make "size" number of cell entries
    for i in range(size):
        samp_index = np.random.choice(list(range(0,len(ex))))   # For sample, pick some wavelength
        
        data_dict['Excitation'].append(ex[samp_index])
        data_dict['Emission'].append(em[samp_index])
        
    # Return just the sample dataframe 
    return (pd.DataFrame(data_dict))


# Run the code outside of defining functions 
if __name__ == "__main__": 
    
    sample_size = 100
    ex_wavelengths = [300,400,500,600]
    em_wavelengths = [350,450,550,650]

    
    result = create_sample(sample_size,ex_wavelengths,em_wavelengths)
    for index, value in enumerate(result['Excitation']):
        print(value,result['Emission'][index])





