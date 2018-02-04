from os import system, path
from site import getusersitepackages

d = getusersitepackages()

if not path.exists(d):
    print('mkdir -p ' + d)
    system('mkdir -p ' + d)

print('cp lib/syngen.py ' + d)
system('cp lib/syngen.py ' + d)
