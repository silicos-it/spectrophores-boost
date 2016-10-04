#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension

from os.path import join as pjoin

# Where to find extensions
SPEC = 'spectrophores'

extensions = []
extensions.append(
    Extension(pjoin(SPEC, 'CRotate'),
              language='c++',
              sources=[pjoin(SPEC, 'CRotate_boost.cpp')],
              libraries=["boost_python"]))


setup(name="PySpectrophore",
      	version='1.0.0',
	  	description='Spectrophore class to be used as Python library',
 		platforms=['Linux', 'Unix'],
   	  	author='Fabio Mendes, Hans de Winter',
   	  	author_email='fabiomendes.farm@gmail.com, hans.dewinter@uantwerpen.be',
      	url='www.uantwerpen.be',
    	ext_modules=extensions, 
		py_modules=[pjoin(SPEC, 'spectrophore')],
		package_dir={'spectrophores': 'spectrophores'},
		packages=['spectrophores']
   )
