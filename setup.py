from setuptools import setup 

setup(
        name = "circuit_metric",
        version = "0.0.0",
        author = "Ben Criger",
        author_email = "BCriger@gmail.com",
        description = ("A little library so you can convert Pauli"
            " error models on quantum circuits into a MWPM metric."),
        packages=['circuit_metric'],
        include_package_data=True
    )