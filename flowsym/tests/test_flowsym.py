#!/usr/bin/env python

"""Tests for `flowsym` package."""

import pytest

# from click.testing import CliRunner
# from flowsym import cli

from flowsym import create_controls, create_sample, measure
import pytest
import pandas as pd
import numpy as np


# import spectrum data
# spectrum_data = pd.read_csv('data/FPbase_Spectra_updated.csv').fillna(value=0)


# Now that we've tested the create sample function, make a fixture for following function tests
@pytest.fixture()
def sample():
    dataframe = create_sample(100)
    return dataframe


# Now that we've tested the create controls function, make a fixture for following function tests
@pytest.fixture()
def controls():
    blue, green, red, far_red, NIR, IR = create_controls(100)
    return blue, green, red, far_red, NIR, IR


def test_create_controls():
    """
    Test the create_controls function to make sure that we created
    all of our control DataFrames correctly
    """

    greens, reds = create_controls(10, ['green', 'red'])  # Params for test function

    assert len(greens) == 10  # Did we output dataframe of correct size?
    assert type(reds) == type(pd.DataFrame())  # Did we output an actual dataframe object?
    assert list(greens.columns) == ['Wavelength', 'Excitation Efficiency',
                                    'Emission Efficiency', 'Copy number']  # Did we make the right columns?

    # Check to make sure wavelengths are equal
    for index, val in greens.iterrows():
        assert val['Wavelength'] == reds['Wavelength'][index]
        assert val['Excitation Efficiency'] != reds['Excitation Efficiency'][index]
        assert val['Emission Efficiency'] != reds['Emission Efficiency'][index]

    # Make sure all excitation or emission efficiencies are the same in a given dataframe
    assert greens['Excitation Efficiency'][0] == greens['Excitation Efficiency'][
        np.random.choice(range(1, len(greens)))]
    assert reds['Emission Efficiency'][0] == reds['Emission Efficiency'][np.random.choice(range(1, len(reds)))]


def test_create_sample():
    """
    Test the create sample function to make sure we made our sample DataFrames correctly
    """
    sample = create_sample(10, ['blue', 'green', 'NIR'])  # Params for test function

    assert len(sample) == 10  # Is dataframe correct size from input step?
    assert type(sample) == type(pd.DataFrame())  # Did we actually output a dataframe?
    assert list(sample.columns) == ['Wavelength', 'Excitation Efficiency', 'Emission Efficiency',
                                    'Copy number']  # Are these the columns?

    # Check to make sure excitation wavelengths aren't the same as emission
    for index, val in sample.iterrows():
        assert val['Excitation Efficiency'] != val['Emission Efficiency']


def test_measure(sample):
    measured = measure(sample)

    assert len(list(measured.columns)) == 6
    assert type(measured) == type(pd.DataFrame())

# @pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
#
#
# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
#
#
# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert 'flowsym.cli.main' in result.output
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert '--help  Show this message and exit.' in help_result.output
