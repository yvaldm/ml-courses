#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:13:11 2019

@author: valery.yakovlev
"""

import numpy as np
from datascience import *

%matplotlib inline

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from urllib.request import urlopen
import re

def read_url(url):
    return re.sub('\\s+', ' ', urlopen(url).read().decode())

little_woman_url = 'http://www.gutenberg.org/cache/epub/514/pg514.txt'
little_woman_text = read_url(little_woman_url)
chapters = little_woman_text.split('CHAPTER ')[1:]
Table().with_column('Text', chapters)

np.char.count(chapters, 'Christmas')
#np.char.count(chapters, 'Jo')

Table().with_columns(        
        'Jo', np.char.count(chapters, 'Jo'),
        'Meg', np.char.count(chapters, 'Meg'),
        'Amy', np.char.count(chapters, 'Amy'),
        'Beth', np.char.count(chapters, 'Beth'),
        'Laurie', np.char.count(chapters, 'Laurie')        
        )

###
### visualization
###

Table().with_columns(
        
        'Jo', np.char.count(chapters, 'Jo'),
        'Meg', np.char.count(chapters, 'Meg'),
        'Amy', np.char.count(chapters, 'Amy'),
        'Beth', np.char.count(chapters, 'Beth'),
        'Laurie', np.char.count(chapters, 'Laurie')
        ).cumsum().plot()

##
##
##

Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of periods', np.char.count(chapters, '.'), 
      ]).scatter('Number of periods')
