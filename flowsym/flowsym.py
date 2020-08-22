"""Main module."""

import numpy as np
import pandas as pd
from fcsy.fcs import write_fcs
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from unidip import UniDip
from copy import deepcopy
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture


spectrum_data = pd.read_csv('data/FPbase_Spectra_updated.csv').fillna(value=0)


def create_controls(size, colors=('blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'far_red', 'nir', 'ir')):
    """
    This is a function that takes a DataFrame size (i.e. number of controls) and
    a list of colors the user wants to run controls for.

    :type size: int
    :param size:
    :type colors: list
    :param colors: 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'far_red', 'NIR', 'IR'
    :return: tuple with final control results
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
    nir_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    ir_em_efficiency = spectrum_data['APC/Cy7 EM']
    cyan_em_efficiency = spectrum_data['CFP EM']
    yellow_em_efficiency = spectrum_data['EYFP EM']
    orange_em_efficiency = spectrum_data['mOrange EM']

    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green': [list(wavelengths), list(green_em_efficiency)],
                     'red': [list(wavelengths), list(red_em_efficiency)],
                     'blue': [list(wavelengths), list(blue_em_efficiency)],
                     'far_red': [list(wavelengths), list(far_red_em_efficiency)],
                     'nir': [list(wavelengths), list(nir_em_efficiency)],
                     'ir': [list(wavelengths), list(ir_em_efficiency)],
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

    for df in final_control_results:
        df['Copy number'] = np.round(np.random.normal(100, size=size, scale=1))

    # Return tuple of the list for easy access of colors
    return tuple(final_control_results)


def create_sample(size, colors=['blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'far_red', 'nir', 'ir'],
                  weights=[]):
    """
    This is a function that takes a defined dataframe length for number of samples (int)
    and excitation and emission wavelengths (list,list). Assumes equal probability of each
    color unless specified by the user.

    :type colors list of strings
    :param colors: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'far_red', 'NIR', 'IR']
    :type weights list
    :param weights (e.g. [0])
    :type size int
    :param size (e.g. 1000 points)
    :return: pandas DataFrame
    """

    # Check to make sure the inputs were of correct type
    if type(size) != int:
        raise TypeError("size cannot be of type: " + str(type(size)))
    elif type(colors) != list:
        raise TypeError("Color list cannot be of type: " + str(type(colors)))

    # Make excitation and emission data easier to read in to dictionary
    wavelengths = spectrum_data['Wavelength']

    green_ex_efficiency = spectrum_data['Fluorescein (FITC) EX']
    red_ex_efficiency = spectrum_data['Kaede (Red) EX']
    blue_ex_efficiency = spectrum_data['Pacific Blue EX']
    far_red_ex_efficiency = spectrum_data['APC (allophycocyanin) AB']
    nir_ex_efficiency = spectrum_data['PerCP-Cy5.5 AB']
    ir_ex_efficiency = spectrum_data['APC/Cy7 EX']
    cyan_ex_efficiency = spectrum_data['CFP EX']
    yellow_ex_efficiency = spectrum_data['EYFP EX']
    orange_ex_efficiency = spectrum_data['mOrange EX']

    # Excitation and emission data for each color control - pre-defined information
    excitation_dict = {'green': [list(wavelengths), list(green_ex_efficiency)],
                       'red': [list(wavelengths), list(red_ex_efficiency)],
                       'blue': [list(wavelengths), list(blue_ex_efficiency)],
                       'far_red': [list(wavelengths), list(far_red_ex_efficiency)],
                       'nir': [list(wavelengths), list(nir_ex_efficiency)],
                       'ir': [list(wavelengths), list(ir_ex_efficiency)],
                       'cyan': [list(wavelengths), list(cyan_ex_efficiency)],
                       'yellow': [list(wavelengths), list(yellow_ex_efficiency)],
                       'orange': [list(wavelengths), list(orange_ex_efficiency)]}

    # Make excitation and emission data easier to read in to dictionary
    green_em_efficiency = spectrum_data['Fluorescein (FITC) EM']
    red_em_efficiency = spectrum_data['Kaede (Red) EM']
    blue_em_efficiency = spectrum_data['Pacific Blue EM']
    far_red_em_efficiency = spectrum_data['APC (allophycocyanin) EM']
    nir_em_efficiency = spectrum_data['PerCP-Cy5.5 EM']
    ir_em_efficiency = spectrum_data['APC/Cy7 EM']
    cyan_em_efficiency = spectrum_data['CFP EM']
    yellow_em_efficiency = spectrum_data['EYFP EM']
    orange_em_efficiency = spectrum_data['mOrange EM']

    # Excitation and emission data for each color control - pre-defined information
    emission_dict = {'green': [list(wavelengths), list(green_em_efficiency)],
                     'red': [list(wavelengths), list(red_em_efficiency)],
                     'blue': [list(wavelengths), list(blue_em_efficiency)],
                     'far_red': [list(wavelengths), list(far_red_em_efficiency)],
                     'nir': [list(wavelengths), list(nir_em_efficiency)],
                     'ir': [list(wavelengths), list(ir_em_efficiency)],
                     'cyan': [list(wavelengths), list(cyan_em_efficiency)],
                     'yellow': [list(wavelengths), list(yellow_em_efficiency)],
                     'orange': [list(wavelengths), list(orange_em_efficiency)]}

    # Set dictionary to be made into DataFrame
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

    # Return just the sample DataFrame
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

    :type lasers: list of int
    :param dataframe:
    :type dataframe: pandas.DataFrame
    :param lasers:
    :type channels list
    :param channels:
    :type create_fcs bool
    :param create_fcs:
    :param outfile_name:
    :return: DataFrame and file
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
        gaussian_dist_of_laser = np.array(np.round(np.random.normal(loc=laser, scale=2.0, size=10000)))
        new_array = [x for x in gaussian_dist_of_laser if laser + 5 >= x >= laser - 5]

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

    # Create new DataFrame and output
    output = pd.DataFrame(new_dataframe_list, columns=column_names)

    if create_fcs:
        write_fcs(output, outfile_name)
        print("FCS file created with filename: " + str(outfile_name))

    return output


def cluster(measured_data, min_cluster_size=50, savefig=True):
    """
    This is a function to cluster flow cytometry data that has been measured in fluorescence channels using
    density-based spatial clustering of applications with noise (DBSCAN), which clusters based on density of points
    in an unsupervised method. The number of clusters does not need to be explicitly stated by the users. The only
    parameter that needs to be optimized is min_cluster_size, which is set to 50 here. But I recommend 1% of the len(
    data) Resulting plots are a bar chart showing the number of cells in each cluster and a heatmap of the median
    fluorescence intensity in each channel for each cluster.

    Note: clusters that are labeled '0' are cells that the DBSCAN could not cluster.

    Returns a tuple of two dictionaries. The first dictionary is the median fluorescence represented in the heatmap
    while the second dictionary holds all the fluorescence vectors for each cluster. Both of these are needed
    for a dip test and re-clustering.

    :rtype: tuple of dict
    :type measured_data file
    :param measured_data
    :type min_cluster_size int
    :param min_cluster_size
    :type savefig bool
    :param savefig
    :return:
    """

    # Create the clustering object
    cluster_obj = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    # Perform the clustering
    cluster_obj.fit(measured_data)

    # Find the number of cells in each cluster
    cluster_counts = {}

    # clusters are
    for i in cluster_obj.labels_:
        if i not in cluster_counts.keys():
            cluster_counts[str(i + 1)] = list(cluster_obj.labels_).count(i)

    X = []

    # Make a 2d array of the vectors
    for index, row in measured_data.iterrows():
        X.append([x for x in row])

    # Make a dictionary for our clusters to hold their associated vectors
    cluster_dict = {}
    for cluster_num in cluster_obj.labels_:
        if cluster_num not in cluster_dict.keys():
            cluster_dict[cluster_num] = []

    # Add the vector in each cluster
    for index, vector in enumerate(X):
        cluster_dict[cluster_obj.labels_[index]].append(vector)

    final_dictionary = {}

    # Make a new dictionary which will have the median value for each channel in the vector for a heatmap downstream
    for key, value in cluster_dict.items():
        median_values = []
        for i in range(len(value[0])):
            median_values.append(np.median([row[i] for row in value]))
            final_dictionary["Cluster " + str(key + 1)] = median_values

    df = pd.DataFrame(final_dictionary, index=list(measured_data.columns))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(df.transpose(), cmap='copper')

    cluster_names = []
    count = []

    for key, value in cluster_counts.items():
        cluster_names.append(key)
        count.append(value)

    y_pos = np.arange(len(cluster_names))

    ax[0].bar(y_pos, count, color='black')
    ax[0].set_xticks(y_pos)
    ax[0].set_xticklabels(cluster_names)
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Cells per cluster')

    ax[1].set_title('Fluorescence profile of clusters')
    ax[1].set_xlabel('Fluorescence channel')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if savefig:
        plt.savefig("preliminary_clustering")

    return (final_dictionary, cluster_dict)


def dip_test(median_FL_data, total_data, alpha=0.05, save_figure=True):
    """
    Perform a Hartigan's dip test to check for unimodality in clusters and splits clusters if bimodality is found.
    This function will take the highest intensity channel for each cluster and
    check for bimodality to correct for errors in clustering similar fluorescencep profiles.
    Changing alpha will alter how stringent the dip test is. A higher alpha will result in higher detection
    of bimodality, but runs a greater risk of false identification. It is important to note
    that this dip test is relatively coarse grained and will not identify very slight populations
    of mixed cells (e.g. 10 orange cells clustered with 1000 red cells)

    Returns an updated clustering of the primary clustering after performing a dip test
    """

    # Create a copy of the dictionary so we can retain the original clustering data
    change_dict = deepcopy(total_data)

    # Make kde plots
    if 'Cluster 0' in median_FL_data.keys():
        fig, ax = plt.subplots(1, len(median_FL_data.keys()) - 1, figsize=(12, 3))

    else:
        fig, ax = plt.subplots(1, len(median_FL_data.keys()), figsize=(12, 3))

    # Keep track of what plot we're on
    i = 0

    # Get the index of the max fluorescence for each cluster
    for key, value in median_FL_data.items():
        cluster_max_FL_index = np.argmax(value)

        # As long as we aren't cluster one, do our dip test and plot
        if int(key[-1]) - 1 != -1:
            search_key = int(key[-1]) - 1

            # Intensity in each cluster where the intensity is max
            dat = [row[cluster_max_FL_index] for row in total_data[search_key]]

            # Do the dip test
            data = np.msort(dat)
            intervals = UniDip(data, alpha=alpha).run()
            print("Performing dip test on cluster " + str(search_key + 1) + " ... ")

            # Show on the graph where the intervals are
            for j in intervals:
                ax[i].axvspan(data[j[0]], data[j[1]], color='lightblue', alpha=0.4)
                for q in j:
                    ax[i].axvline(data[q], color='red')

            # Split the clusters that failed the dip test into separate clusters
            if len(intervals) > 1:
                split_point = int(np.mean([intervals[0][1], intervals[1][0]]))
                clust1 = data[:split_point]
                clust2 = data[split_point:]

                # Reset current cluster number to cluster 1 and make a new cluster to the dictionary
                print("Identified bimodality in cluster " + str(search_key + 1) + ", reclustering data ... ")
                change_dict[max(total_data.keys()) + 1] = [row for row in total_data[search_key] if
                                                           row[cluster_max_FL_index] in clust2]
                change_dict[search_key] = [row for row in total_data[search_key] if row[cluster_max_FL_index] in clust1]

            # Plot data
            sns.kdeplot(data, ax=ax[i], color='black')

            ax[i].set(title='Cluster ' + str(search_key + 1), xlabel='FL ' + str(cluster_max_FL_index + 1), yticks=[])

            # Move to the next plot
            i += 1

        plt.tight_layout()

        # save first figure of the dip test
        if save_figure:
            plt.savefig("Dip_test_example")

        final_reclustered = {}

    # Make a new dictionary which will have the median value for each channel in the vector for a heatmap downstream
    for key, value in change_dict.items():
        med_values = []
        for i in range(len(value[0])):
            med_values.append(np.median([row[i] for row in value]))
            final_reclustered["Cluster " + str(key + 1)] = med_values

    search = np.random.choice(list(median_FL_data.keys()))

    cols = ['FL' + str(i + 1) for i in range(len(median_FL_data[search]))]

    # Dataframe to create heatmap
    reclustered_df = pd.DataFrame(final_reclustered, index=cols)

    # Counts dictionary for barchart
    reclustered_counts = {}

    for key, value in change_dict.items():
        reclustered_counts[key] = len(value)

        # Replot the new clusters
    print("Plotting reclustered data ...")

    fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(reclustered_df.transpose(), cmap='copper')

    reclust = []
    recount = []

    for key, value in reclustered_counts.items():
        reclust.append(int(key) + 1)
        recount.append(value)

    rey_pos = np.arange(len(reclust))

    ax[0].bar(rey_pos, recount, color='black')
    ax[0].set_xticks(rey_pos)
    ax[0].set_xticklabels(reclust)
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Cells per cluster')

    ax[1].set_title('Fluorescence profile of clusters')
    ax[1].set_xlabel('Fluorescence channel')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_figure:
        plt.savefig("reclustered_after_dip_test")

    return change_dict


def gaus_recluster(median_FL_data, total_data, tolerance=.25, savefig=True):
    """
    Applies a gaussian mixture model with n_components=2
    to try and separate rare populations of cells from
    the original clustering. This will apply the model
    and then conduct a Kolmogorov-Smirnov 2 sample test
    to assess significant differences in distributions of
    the split clusters. Two criteria are used to determine
    whether a cluster is saved as split, or if it is preserved
    as it originally was:

    P-value of Ks2 test: If p-value is below 1e-10

    Difference in cluster size: if a cluster is split
    and the difference between the sizes of the new clusters
    is greater than <tolerance> of the total cells in the original
    cluster.

    parameters:
    median_FL_data - data with median FL for each cluster

    total_data - data with all measured FL for each cluster

    tolerance - how different do the sizes of clusters have to
    be before they are considered actually distinct spectrally?
    Increase this to be more stringent in splitting clusters.
    Decrease the value to allow more reclustering at the cost of
    false positives.

    savefig - save figures

    returns:
    reclustered - reclustered dataset of all cells analyzed

    """
    index_max = {}

    # Get the max FL channel index for each cluster that is not 0 (i.e. unclustered)
    for key, value in median_FL_data.items():
        if key[-1] != '0':
            index_max[int(key[-1]) - 1] = np.argmax(value)

    fig, ax = plt.subplots(1, len(list(index_max.keys())), figsize=(12, 3))

    # create a copy of the input data to preserve new and old datasets
    reclustered = deepcopy(total_data)

    i = 0
    for key, value in total_data.items():
        if key != -1:

            # Max fluorescence channel for each cluster
            max_channel = index_max[key]

            # Apply a gaussian mixture model and split into 2 components
            gmm = GaussianMixture(n_components=2)
            gmm.fit(value)

            # Label each cell in our clusters with the label for how they split
            labels = gmm.predict(value)

            # Create a dataframe of the intensity vectors and their new cluster after the split
            frame = pd.DataFrame(value)
            frame['cluster'] = labels

            # subset dataframe based on new cluster number
            pre_clust1 = frame[frame['cluster'] == 0]
            pre_clust2 = frame[frame['cluster'] == 1]

            # Remove the cluster column. Probably redundant to do things this way
            clust1 = pre_clust1[pre_clust1.columns[:-1]]
            clust2 = pre_clust2[pre_clust2.columns[:-1]]

            # Do a ks 2 test to see if clusters are different
            result = ks_2samp(clust1[max_channel], clust2[max_channel])

            # Test how different our cluster populations are. If the difference between the sizes is more than <tolerance>, of the
            # total, then we'll say we actually found a bimodal population to split
            clust_split = abs(len(clust1) - len(clust2)) / (len(clust1) + len(clust2))

            # Keep the split clusters if they meet our splitting criteria, otherwise retain original clusters from DB scan
            if clust_split > tolerance:
                if result[1] < 1e-10:
                    new_val = clust1.values.tolist()
                    new_val2 = clust2.values.tolist()

                    reclustered[key] = new_val

                    reclustered[max(total_data.keys()) + 1] = new_val2

            # Provide kde plots of the distributions to show which ones might
            sns.kdeplot(clust1[max_channel], ax=ax[i], color='crimson')
            sns.kdeplot(clust2[max_channel], ax=ax[i], color='navy')
            ax[i].get_legend().remove()
            ax[i].set_title("Cluster " + str(key + 1) + ' split')

            i += 1

    plt.tight_layout()

    if savefig:
        plt.savefig('gaus_mix_cluster_split')

    final_reclustered = {}

    # Make a new dictionary which will have the median value for each channel in the vector for a heatmap downstream
    for key, value in reclustered.items():
        med_values = []
        for i in range(len(value[0])):
            med_values.append(np.median([row[i] for row in value]))
            final_reclustered["Cluster " + str(key + 1)] = med_values

    # Create a list of column names of the vector (FL1-6)
    search = np.random.choice(list(median_FL_data.keys()))
    cols = ['FL' + str(i + 1) for i in range(len(median_FL_data[search]))]

    # Dataframe to create heatmap
    reclustered_df = pd.DataFrame(final_reclustered, index=cols)

    # Counts dictionary for barchart
    reclustered_counts = {}

    for key, value in reclustered.items():
        reclustered_counts[key] = len(value)

    # Replot the new clusters
    print("Plotting reclustered data ...")

    fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(reclustered_df.transpose(), cmap='copper')

    reclust = []
    recount = []

    for key, value in reclustered_counts.items():
        reclust.append(int(key) + 1)
        recount.append(value)

    rey_pos = np.arange(len(reclust))

    ax[0].bar(rey_pos, recount, color='black')
    ax[0].set_xticks(rey_pos)
    ax[0].set_xticklabels(reclust)
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Cells per cluster')

    ax[1].set_title('Fluorescence profile of clusters')
    ax[1].set_xlabel('Fluorescence channel')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if savefig:
        plt.savefig('reclustered_after_gaus_mix_ks2')

    return reclustered
