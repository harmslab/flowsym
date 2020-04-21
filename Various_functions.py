# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:57:16 2020

@author: Michael
"""
import numpy as np
import pandas as pd
#from __init__ import spectrum_data

# Temp - init file in same directory screws up pytest run
spectrum_data= pd.read_csv('https://raw.githubusercontent.com/mshavlik/FACSimulator/master/FPbase_Spectra.csv').fillna(value=0)



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
        controls_dict[i] = {'Wavelength':[],'Excitation Efficiency':[], 'Emission Efficiency':[]}
        
        
        
    # Make excitation and emission data easier to read in to dictionary
    wavelengths = spectrum_data['Wavelength']
    
    green_ex_efficiency = spectrum_data['Fluorescein (FITC) EX']
    red_ex_efficiency = spectrum_data['Kaede (Red) EX']
    blue_ex_efficiency = spectrum_data['Pacific Blue EX']
    far_red_ex_efficiency = spectrum_data['APC (allophycocyanin)']
    NIR_ex_efficiency = spectrum_data['PerCP-Cy5.5']
    IR_ex_efficiency = spectrum_data['APC/Cy7 EX']
    
    
    
    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green':[list(wavelengths),list(green_ex_efficiency)],'red':[list(wavelengths),list(red_ex_efficiency)],
                    'blue':[list(wavelengths),list(blue_ex_efficiency)],'far_red':[list(wavelengths),list(far_red_ex_efficiency)],
                    'nir':[list(wavelengths),list(NIR_ex_efficiency)],'ir':[list(wavelengths),list(IR_ex_efficiency)]}
    
    
    
    # Make excitation and emission data easier to read in to dictionary
    green_em_efficiency = spectrum_data['Fluorescein (FITC) EM']
    red_em_efficiency = spectrum_data['Kaede (Red) EM']
    blue_em_efficiency = spectrum_data['Pacific Blue EM']
    far_red_em_efficiency = spectrum_data['APC (allophycocyanin) EM']
    NIR_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    IR_em_efficiency = spectrum_data['APC/Cy7 EM']
    
    
    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green':[list(wavelengths),list(green_em_efficiency)],'red':[list(wavelengths),list(red_em_efficiency)],
                    'blue':[list(wavelengths),list(blue_em_efficiency)],'far_red':[list(wavelengths),list(far_red_em_efficiency)],
                    'nir':[list(wavelengths),list(NIR_em_efficiency)],'ir':[list(wavelengths),list(IR_em_efficiency)]}
    
    
    # Match colors that the user wants to excitation and emission data
    for key, value in controls_dict.items():
        key = key.lower()     # Doesn't matter if user entered capital letters or not
        if key in excitation_dict:
            value['Wavelength'] = [excitation_dict[key][0]]
            value['Excitation Efficiency'] = [excitation_dict[key][1]]
            value['Emission Efficiency'] = [emission_dict[key][1]]
            
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
    
    # Make excitation and emission data easier to read in to dictionary
    wavelengths = spectrum_data['Wavelength']
    
    green_ex_efficiency = spectrum_data['Fluorescein (FITC) EX']
    red_ex_efficiency = spectrum_data['Kaede (Red) EX']
    blue_ex_efficiency = spectrum_data['Pacific Blue EX']
    far_red_ex_efficiency = spectrum_data['APC (allophycocyanin)']
    NIR_ex_efficiency = spectrum_data['PerCP-Cy5.5']
    IR_ex_efficiency = spectrum_data['APC/Cy7 EX']
    
    
    
    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green':[list(wavelengths),list(green_ex_efficiency)],'red':[list(wavelengths),list(red_ex_efficiency)],
                    'blue':[list(wavelengths),list(blue_ex_efficiency)],'far_red':[list(wavelengths),list(far_red_ex_efficiency)],
                    'nir':[list(wavelengths),list(NIR_ex_efficiency)],'ir':[list(wavelengths),list(IR_ex_efficiency)]}
    
    
    # Make excitation and emission data easier to read in to dictionary
    green_em_efficiency = spectrum_data['Fluorescein (FITC) EM']
    red_em_efficiency = spectrum_data['Kaede (Red) EM']
    blue_em_efficiency = spectrum_data['Pacific Blue EM']
    far_red_em_efficiency = spectrum_data['APC (allophycocyanin) EM']
    NIR_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    IR_em_efficiency = spectrum_data['APC/Cy7 EM']
    
    
    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green':[list(wavelengths),list(green_em_efficiency)],'red':[list(wavelengths),list(red_em_efficiency)],
                    'blue':[list(wavelengths),list(blue_em_efficiency)],'far_red':[list(wavelengths),list(far_red_em_efficiency)],
                    'nir':[list(wavelengths),list(NIR_em_efficiency)],'ir':[list(wavelengths),list(IR_em_efficiency)]}
    
    # Set dictionary to be made into dataframe
    sample_dict = {'Wavelength':[],'Excitation Efficiency':[], 'Emission Efficiency':[]}
    
    # Make "size" number of cell entries
    for i in range(size):
        color_to_pick = np.random.choice(colors).lower()
        
        sample_dict['Wavelength'].append(excitation_dict[color_to_pick][0])
        sample_dict['Excitation Efficiency'].append(excitation_dict[color_to_pick][1])
        sample_dict['Emission Efficiency'].append(emission_dict[color_to_pick][1])
        
    # Return just the sample dataframe 
    return (pd.DataFrame(sample_dict))





# Bandwidth on lasers is +-5 nm. channels are [450+-25, 525+-25, 600+-30, 665+-15, 720+-30, 785+-30] for filter set 2
def measure(dataframe,lasers=[405,488,561,638],channels=[1,2,3,4,5,6]):
    """
    This is a function that will measure fluorescence intensity for any given sample
    dataframe and laser/channel parameters. Output will be a new dataframe with
    len(input_dataframe) entries and fluorescence intensity values for each channel
    """
    # Bandwidth for each fluorescence channel
    channels_information = {1:list(range(425,475)),2:list(range(500,550)),3:list(range(570,630)),4:list(range(650,680)),
                            5:list(range(690,750)),6:list(range(755,805))}


    # This is the list that will hold all of the intensity vectors for each cell
    new_dataframe_list = [['FL'+str(i) for i in channels]]
    
    # where are our laser wavelengths in our input dataframe?
    laser_indices = []
    
    # For each laser, find the indices for their wavelengths so we can get excitation efficiencies later
    for laser in lasers:
        for index2, wave in enumerate(dataframe['Wavelength'][0]):
            if wave in list(range(laser-5,laser+5)):
                laser_indices.append(index2)
                
    # for each cell that is being analyzed
    for index, row in dataframe.iterrows():
        intensity_vector = []
                
        # Calculate peak excitation efficiency for our cell given all lasers at once (collinear laser set up)
        excitation_max = max([row['Excitation Efficiency'][x] for x in laser_indices])
        
        # Define an amplification for the signal, add some noise to it
        amplification = np.random.normal(10**3,size=1) * np.random.uniform(0.75,1.25)
        amplified_signal = amplification*excitation_max
            
        
        
        # For each fluorescence channel, find the appropriate emission efficiencies for given wavelengths
        for channel in channels:
            channel_indices = []
            em_chan = channels_information[channel]
            for index2, wave in enumerate(row['Wavelength']):
                if wave in em_chan:
                    channel_indices.append(index2)
            
            # Calculate an emmision intensity based on noise, emission efficiency, and the excitation signal
            emission_intensity = max([row['Emission Efficiency'][x] for x in channel_indices]) * amplified_signal * np.random.uniform(3,size=1)
            intensity_vector.append(float(emission_intensity))
    

    
        new_dataframe_list.append(intensity_vector)
        
    column_names = new_dataframe_list.pop(0)

    # Create new dataframe and output
    output = pd.DataFrame(new_dataframe_list,columns=column_names)
    
    return output

# Run the code outside of defining functions 
if __name__ == "__main__": 
    
    sample_size = 100
    
    sample = create_sample(sample_size,['green','red'])
    
    measurements = measure(sample)
    
    




