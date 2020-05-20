# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:57:16 2020

:author: Michael
:author: Luis
"""

# Check to see if fcsy is installed on machine 
import importlib.util

package_name = 'fcsy'
spec = importlib.util.find_spec(package_name)
if spec is None:
    print(
        f"{package_name} is not installed, please install {package_name} to write fcs files in the \'measure\' function!")

import numpy as np
import pandas as pd
import time
from fcsy.fcs import write_fcs

# from __init__ import spectrum_data

# Temp - init file in same directory screws up pytest run
spectrum_data = pd.read_csv(
    'data/FPbase_Spectra_updated.csv').fillna(value=0)


def create_controls(size, colors=['Blue', 'Cyan', 'Green', 'Yellow', 'Orange', 'Red', 'Far_red', 'NIR', 'IR']):
    """
    This is a function that takes a DataFrame size (i.e. number of controls) and
    a list of colors the user wants to run controls for.
    :param size:
    :param colors:
    :return:
    """

    # Check to make sure the inputs were of correct type 
    if type(size) != int:
        raise TypeError("size cannot be of type: " + str(type(size)))
    elif type(colors) != list:
        raise TypeError("Ex cannot be of type: " + str(type(colors)))

    # Controls data - accept whatever colors the user provides
    controls_dict = {}
    for i in colors:
        controls_dict[i] = {'Wavelength': [], 'Excitation Efficiency': [], 'Emission Efficiency': []}

    # Make excitation and emission data easier to read in to dictionary
    wavelengths = spectrum_data['Wavelength']

    green_ex_efficiency = spectrum_data['Fluorescein (FITC) EX']
    red_ex_efficiency = spectrum_data['Kaede (Red) EX']
    blue_ex_efficiency = spectrum_data['Pacific Blue EX']
    far_red_ex_efficiency = spectrum_data['APC (allophycocyanin) AB']
    NIR_ex_efficiency = spectrum_data['PerCP-Cy5.5 AB']
    IR_ex_efficiency = spectrum_data['APC/Cy7 EX']
    cyan_ex_efficiency = spectrum_data['CFP EX']
    yellow_ex_efficiency = spectrum_data['EYFP EX']
    orange_ex_efficiency = spectrum_data['mOrange EX']

    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green': [list(wavelengths), list(green_ex_efficiency)],
                       'red': [list(wavelengths), list(red_ex_efficiency)],
                       'blue': [list(wavelengths), list(blue_ex_efficiency)],
                       'far_red': [list(wavelengths), list(far_red_ex_efficiency)],
                       'nir': [list(wavelengths), list(NIR_ex_efficiency)],
                       'ir': [list(wavelengths), list(IR_ex_efficiency)],
                       'cyan': [list(wavelengths), list(cyan_ex_efficiency)],
                       'yellow': [list(wavelengths), list(yellow_ex_efficiency)],
                       'orange': [list(wavelengths), list(orange_ex_efficiency)]}

    # Make excitation and emission data easier to read in to dictionary
    green_em_efficiency = spectrum_data['Fluorescein (FITC) EM']
    red_em_efficiency = spectrum_data['Kaede (Red) EM']
    blue_em_efficiency = spectrum_data['Pacific Blue EM']
    far_red_em_efficiency = spectrum_data['APC (allophycocyanin) EM']
    NIR_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    IR_em_efficiency = spectrum_data['APC/Cy7 EM']
    cyan_em_efficiency = spectrum_data['CFP EM']
    yellow_em_efficiency = spectrum_data['EYFP EM']
    orange_em_efficiency = spectrum_data['mOrange EM']

    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green': [list(wavelengths), list(green_em_efficiency)],
                     'red': [list(wavelengths), list(red_em_efficiency)],
                     'blue': [list(wavelengths), list(blue_em_efficiency)],
                     'far_red': [list(wavelengths), list(far_red_em_efficiency)],
                     'nir': [list(wavelengths), list(NIR_em_efficiency)],
                     'ir': [list(wavelengths), list(IR_em_efficiency)],
                     'cyan': [list(wavelengths), list(cyan_em_efficiency)],
                     'yellow': [list(wavelengths), list(yellow_em_efficiency)],
                     'orange': [list(wavelengths), list(orange_em_efficiency)]}

    # Match colors that the user wants to excitation and emission data
    for key, value in controls_dict.items():
        key = key.lower()  # Doesn't matter if user entered capital letters or not
        if key in excitation_dict:
            value['Wavelength'] = [excitation_dict[key][0]]
            value['Excitation Efficiency'] = [excitation_dict[key][1]]
            value['Emission Efficiency'] = [emission_dict[key][1]]

        else:
            raise NameError(str(key) + ' is not an available control, try: ' +
                            str(list(excitation_dict.keys())))

    # Create a new dictionary that will keep the associated colors with dataframe objects
    results_dict = {}
    for key, value in controls_dict.items():
        results_dict[key] = pd.DataFrame(value)

    # Finally, create a list that will hold all DFs while preserving color order
    final_control_results = []
    for i in colors:
        final_control_results.append(pd.concat([results_dict[i]] * size, ignore_index=True))

    # Return tuple of the list for easy access of colors
    return tuple(final_control_results)


def create_sample(size, colors=['Blue', 'Cyan', 'Green', 'Yellow', 'Orange', 'Red', 'Far_red', 'NIR', 'IR'],
                  weights=[]):
    """
    This is a function that takes a defined dataframe length for number of samples (int)
    and excitation and emission wavelengths (list,list). Assumes equal probability of each
    color unless specified by the user
    :param size:
    :param colors:
    :param weights:
    :return:
    """


    # Check to make sure the inputs were of correct type 
    if type(size) != int:
        raise TypeError("size cannot be of type: " + str(type(size)))
    elif type(colors) != list:
        raise TypeError("Ex cannot be of type: " + str(type(colors)))

    # Make excitation and emission data easier to read in to dictionary
    wavelengths = spectrum_data['Wavelength']

    green_ex_efficiency = spectrum_data['Fluorescein (FITC) EX']
    red_ex_efficiency = spectrum_data['Kaede (Red) EX']
    blue_ex_efficiency = spectrum_data['Pacific Blue EX']
    far_red_ex_efficiency = spectrum_data['APC (allophycocyanin) AB']
    NIR_ex_efficiency = spectrum_data['PerCP-Cy5.5 AB']
    IR_ex_efficiency = spectrum_data['APC/Cy7 EX']
    cyan_ex_efficiency = spectrum_data['CFP EX']
    yellow_ex_efficiency = spectrum_data['EYFP EX']
    orange_ex_efficiency = spectrum_data['mOrange EX']

    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green': [list(wavelengths), list(green_ex_efficiency)],
                       'red': [list(wavelengths), list(red_ex_efficiency)],
                       'blue': [list(wavelengths), list(blue_ex_efficiency)],
                       'far_red': [list(wavelengths), list(far_red_ex_efficiency)],
                       'nir': [list(wavelengths), list(NIR_ex_efficiency)],
                       'ir': [list(wavelengths), list(IR_ex_efficiency)],
                       'cyan': [list(wavelengths), list(cyan_ex_efficiency)],
                       'yellow': [list(wavelengths), list(yellow_ex_efficiency)],
                       'orange': [list(wavelengths), list(orange_ex_efficiency)]}

    # Make excitation and emission data easier to read in to dictionary
    green_em_efficiency = spectrum_data['Fluorescein (FITC) EM']
    red_em_efficiency = spectrum_data['Kaede (Red) EM']
    blue_em_efficiency = spectrum_data['Pacific Blue EM']
    far_red_em_efficiency = spectrum_data['APC (allophycocyanin) EM']
    NIR_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    IR_em_efficiency = spectrum_data['APC/Cy7 EM']
    cyan_em_efficiency = spectrum_data['CFP EM']
    yellow_em_efficiency = spectrum_data['EYFP EM']
    orange_em_efficiency = spectrum_data['mOrange EM']

    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green': [list(wavelengths), list(green_em_efficiency)],
                     'red': [list(wavelengths), list(red_em_efficiency)],
                     'blue': [list(wavelengths), list(blue_em_efficiency)],
                     'far_red': [list(wavelengths), list(far_red_em_efficiency)],
                     'nir': [list(wavelengths), list(NIR_em_efficiency)],
                     'ir': [list(wavelengths), list(IR_em_efficiency)],
                     'cyan': [list(wavelengths), list(cyan_em_efficiency)],
                     'yellow': [list(wavelengths), list(yellow_em_efficiency)],
                     'orange': [list(wavelengths), list(orange_em_efficiency)]}

    # Set dictionary to be made into dataframe
    sample_dict = {'Wavelength': [], 'Excitation Efficiency': [], 'Emission Efficiency': []}

    # Make "size" number of cell entries
    for i in range(size):
        if len(weights) == 0:
            color_to_pick = np.random.choice(colors).lower()
        else:
            color_to_pick = np.random.choice(colors, p=weights).lower()

        sample_dict['Wavelength'].append(excitation_dict[color_to_pick][0])
        sample_dict['Excitation Efficiency'].append(excitation_dict[color_to_pick][1])
        sample_dict['Emission Efficiency'].append(emission_dict[color_to_pick][1])

    data = pd.DataFrame(sample_dict)

    # Create protein copy number
    copies = np.round(np.random.normal(100, size=len(data), scale=20))
    data['Copy number'] = copies

    # Return just the sample dataframe 
    return data


# Bandwidth on lasers is +-5 nm. channels are [450+-25, 525+-25, 600+-30, 665+-15, 720+-30, 785+-30] for filter set 2
def measure(dataframe, lasers=[405, 488, 561, 638], channels=[1, 2, 3, 4, 5, 6],
            create_fcs=True, outfile_name='data/sample_output.fcs'):
    """
    This is a function that will measure fluorescence intensity for any given sample
    DataFrame and laser/channel parameters. Output will be an fcs file (default) that is
    the same size as the sample you ran in the function. Alternatively, you can return
    just a pandas DataFrame object by setting return_fcs=False. The user can set the output
    file name manually to simulate creating multiple samples and measurements.
    """
    # Bandwidth for each fluorescence channel
    channels_information = {1: list(range(425, 475)), 2: list(range(500, 550)), 3: list(range(570, 630)),
                            4: list(range(650, 680)),
                            5: list(range(690, 750)), 6: list(range(755, 805))}

    # This is the list that will hold all of the intensity vectors for each cell
    new_dataframe_list = [['FL' + str(i) for i in channels]]

    # where are our laser wavelengths in our input dataframe?
    laser_indices = {}

    # For each laser, find the indices for their wavelengths and their gaussian efficiencies 
    for laser in lasers:
        # This part makes a gaussian distribution of each laser+-5
        counts_dict = {}
        myarray = np.array(np.round(np.random.normal(loc=laser, scale=2.0, size=10000)))
        new_array = [x for x in myarray if laser + 5 >= x >= laser - 5]

        for i in new_array:
            if i not in counts_dict.keys():
                counts_dict[i] = list(new_array).count(i)

        max_count = max(counts_dict.values())

        for key, value in counts_dict.items():
            counts_dict[key] = value / max_count

        # Find the wavelength indices that our lasers hit - make a dictionary with indices as keys and laser
        # efficiencies as values
        for index2, wave in enumerate(dataframe['Wavelength'][0]):
            if wave in counts_dict.keys():
                laser_indices[index2] = counts_dict[wave]

    # figure out unique emission profiles based on color so we know when to end the loop
    copy = dataframe.copy()
    copy['Emission Efficiency'] = copy['Emission Efficiency'].astype(str)

    # Create numpy arrays to randomly sample from based on the number of excited molecules
    emission_reference = {}

    for index, row in dataframe.iterrows():
        if str(row['Emission Efficiency']) not in emission_reference.keys():
            waves_to_add = np.array([round(value * 100) * [row['Wavelength'][index]] for index, value in
                                     enumerate(row['Emission Efficiency']) if value >= 0.01])
            emission_reference[str(row['Emission Efficiency'])] = np.array([y for x in waves_to_add for y in x])

        if len(emission_reference.keys()) == len(copy['Emission Efficiency'].unique()):
            break

    # for each cell that is being analyzed
    for index, row in dataframe.iterrows():
        intensity_vector = []

        # Calculate peak excitation efficiency for our cell given all lasers at once (collinear laser set up)
        excitation_max = max([row['Excitation Efficiency'][key] * value for key, value in laser_indices.items()])
        num_excited_proteins = round(row['Copy number'] * excitation_max)

        # Sample emission at wavelengths corresponding to real emission efficiency from FPbase, size=number of
        # excited proteins
        real_emission_wavelengths = np.random.choice(emission_reference[str(row['Emission Efficiency'])],
                                                     size=num_excited_proteins)

        # amp = np.random.choice(list(range(1000,1700,40)))
        # For each fluorescence channel, find the appropriate emission values
        for channel in channels:
            em_chan = channels_information[channel]

            # Find intensity in each channel - NOTE using intersection and set here speed up the code DRAMATICALLY
            emission_intensity = len(set(real_emission_wavelengths).intersection(em_chan)) * (
                        1000 + np.random.normal(0, scale=50))  # Average amplification +- noise

            # add intensity in each channel to the vector
            intensity_vector.append(float(emission_intensity))

        new_dataframe_list.append(intensity_vector)

    column_names = new_dataframe_list.pop(0)

    # Create new dataframe and output
    output = pd.DataFrame(new_dataframe_list, columns=column_names)

    if create_fcs:
        write_fcs(output, outfile_name)
        print("FCS file created with filename: " + str(outfile_name))

    return output


# Run the code outside of defining functions
if __name__ == "__main__":
    sample_size = 1000

    sample = create_sample(sample_size)

    start = time.time()
    measurements = measure(sample)
    stop = time.time()
    print("Time to run measure was " + str(round(stop - start, 3)) + " seconds")
