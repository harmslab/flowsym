# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:57:16 2020

@author: Michael
"""
import numpy as np
import pandas as pd



def create_controls(size,colors=['Blue','Green','Red','Far_red','NIR','IR']):
    """
    This is a function that takes a dataframe size (i.e. number of controls) and 
    a list of colors the user wants to run controls for
    """
    
    # Check to make sure the inputs were of correct type 
    if type(size) != int:
        raise TypeError("size cannot be of type: "+str(type(size)))
    elif type(colors) != list:
        raise TypeError("Ex cannot be of type: "+str(type(colors)))
            
        
    # Controls data - accept whatever colors the user provides
    controls_dict = {}
    for i in colors:
        controls_dict[i] = {'Excitation':[],'Emission':[]}
    
    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green':[list(range(435,525))],'red':[list(range(450,588))],
                    'blue':[list(range(336,440))],'far_red':[list(range(540,673))],
                    'nir':[list(range(560,722))],'ir':[list(range(540,810))]}
    
    emission_dict = {'green':[list(range(492,590))],'red':[list(range(555,640))],
                    'blue':[list(range(425,530))],'far_red':[list(range(623,740))],
                    'nir':[list(range(655,815))],'ir':[list(range(731,845))]}
    
    
    # Match colors that the user wants to excitation and emission data
    for key, value in controls_dict.items():
        key = key.lower()     # Doesn't matter if user entered capital letters or not
        if key in excitation_dict:
            value['Excitation'] = excitation_dict[key]
            value['Emission'] = emission_dict[key]
        else:
            raise NameError(str(key) + ' is not an available control, try: ' + 
                            str(list(excitation_dict.keys())))
       
    # Create a new dictionary that will keep the associated colors with dataframe objects
    results_dict = {}
    for key,value in controls_dict.items():
        results_dict[key] = pd.DataFrame(value)
    
    
    # Finally, create a list that will hold all DFs while preserving color order
    final_control_results = []
    for i in colors:
        final_control_results.append(pd.concat([results_dict[i]]*size,ignore_index=True))

    # Return tuple of the list for easy access of colors
    return tuple(final_control_results)






def create_sample(size,colors=['Blue','Green','Red','Far_red','NIR','IR']):
    """
    This is a function that takes a defined dataframe length for number of samples (int)
    and excitation and emission wavelengths (list,list)
    """
    
    # Check to make sure the inputs were of correct type 
    if type(size) != int:
        raise TypeError("size cannot be of type: "+str(type(size)))
    elif type(colors) != list:
        raise TypeError("Ex cannot be of type: "+str(type(colors)))
    
        
        
    # Excitation and emission data for each color - pre-defined information
    excitation_dict = {'green':list(range(435,525)),'red':list(range(450,588)),
                       'blue':list(range(336,440)),'far_red':list(range(540,673)),
                       'nir':list(range(560,722)),'ir':list(range(540,810))}
    
    emission_dict = {'green':list(range(492,590)),'red':list(range(555,640)),
                     'blue':list(range(425,530)),'far_red':list(range(623,740)),
                     'nir':list(range(655,815)),'ir':list(range(731,845))}
    
    # Set dictionary to be made into dataframe
    sample_dict = {'Excitation': [],'Emission': []}
    
    # Make "size" number of cell entries
    for i in range(size):
        color_to_pick = np.random.choice(colors).lower()
        
        sample_dict['Excitation'].append(excitation_dict[color_to_pick])
        sample_dict['Emission'].append(emission_dict[color_to_pick])
        
    # Return just the sample dataframe 
    return (pd.DataFrame(sample_dict))






# Run the code outside of defining functions 
if __name__ == "__main__": 
    
    sample_size = 100
    
    sample = create_sample(sample_size,['green','red'])
    red,green = create_controls(sample_size,colors=['Red','Green'])
    
    print(sample)




