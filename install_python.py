from sys import path
from os import system
directories = [p for p in path if 'site-packages' in p]
if len(directories) == 0:
  print('Could not find python packages directory!')
  exit(1)
system('cp lib/syngen.py ' + directories[0])
