import setuptools

with open("README", "r") as fh:
    long_description = fh.read()
    
tft_dependencies = [
           'tensorflow==1.10.0',
           'numpy==1.14.4',
           'matplotlib==2.2.3',
           'scipy==1.1.0',
           'scikit-image==0.14.0',
           'lime==0.1.1.31',
           'pandas',
           'keras==2.2.4',
           'sklearn',
           'flask',
           'flask-cors']

setuptools.setup(
    name="tft",
    version="tft_v1.5",
    author="TFTAuthors",
    author_email="TFTAuthors",
    description="tft-->Testing For Trust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: TBD",
        "Operating System :: OS Independent",
    ],
    install_requires=tft_dependencies,
)