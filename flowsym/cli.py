"""Console script for flowsym."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for flowsym."""
    click.echo("Welcome to FlowSym, a Python API used to simulate flow "
               "cytometry data!"
               "\n\n"
               "If you don't see any warning messages above then you're "
               "all set!")
    click.echo("See documentation at ")
    return 0
# Check to see if fcsy and hdbscan are installed on machine
import importlib.util

package_name = 'fcsy'
spec = importlib.util.find_spec(package_name)
if spec is None:
    print(
        f"{package_name} is not installed, please install {package_name} to write fcs files in the \'measure\' function!")

package2_name = 'hdbscan'
spec2 = importlib.util.find_spec(package2_name)
if spec2 is None:
    print(
        f"{package2_name} is not installed, please install {package2_name} to cluster data in the \'cluster\' function!")

package3_name = 'unidip'
spec3 = importlib.util.find_spec(package3_name)
if spec3 is None:
    print(
        f"{package3_name} is not installed, please install {package3_name} to perform a dip test in the \'dip_test\' "
        f"function!")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
