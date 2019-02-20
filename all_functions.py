# Import libraries

import time
from datetime import datetime as dt
import pandas as pd
import numpy as np
import re
import sys
import operator
import os
import random
import json
try:
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	from matplotlib.legend_handler import HandlerBase
	from matplotlib.widgets import CheckButtons
except:
	pass

from IPython.display import display
from custom_functions import *

pd.set_option('max_colwidth',-1)
pd.set_option('max_columns',999)
pd.set_option('max_rows',999)
#import ipaddress    # Easiest to get using Python 3.x
#import Tkinter as tk
#from Tkinter import filedialog
