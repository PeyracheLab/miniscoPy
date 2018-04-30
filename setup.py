from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np


# compile with:     python setup.py build_ext -i
# clean up with:    python setup.py clean --all
ext_modules = [Extension("miniscopy.cnmfe.oasis",
                         sources=["miniscopy/cnmfe/oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++")]

setup(
    name='miniscopy',
    version='0.1',
    author='Guillaume Viejo',
    author_email='guillaume.viejo@gmail.com',
    # url='https://github.com/simonsfoundation/CaImAn',
    license='GPL-2',
    description='CNMF-E for miniscope data',
    long_description='''
    	simplification of the caiman package https://github.com/flatironinstitute/CaImAn
    	for internal use of the peyrache lab 
        I removed all the test functions and few stuffs
        Trying to keep the minimum
        Goal : to merge it with Neuroseries
    ''',
    
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification miniscope',
    packages=find_packages(exclude=['use_cases', 'use_cases.*']),
    data_files=[('', ['LICENSE.md']),
                ('', ['README.md'])],
    install_requires=[''],
    ext_modules=cythonize(ext_modules)

)