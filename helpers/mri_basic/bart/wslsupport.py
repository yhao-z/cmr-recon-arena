"""Windows-to-WSL path translation used by bart.py: rewrites drive-letter
paths (e.g. c:/tmp) to their WSL mount form (/mnt/c/tmp).
Local changes are marked with ``MV``.
"""

import string
import os

def PathCorrection(inData):
	"""Convert a Windows path to the corresponding WSL path."""
	outData=inData
	for i in string.ascii_lowercase: #Replace drive letters with /mnt/
		outData=outData.replace(i+':','/mnt/'+i) #if drive letter is supplied in lowercase
		outData=outData.replace(i.upper()+':','/mnt/'+i) #if drive letter is supplied as uppercase
	outData=outData.replace(os.path.sep, '/') #Change windows filesep to linux filesep

	return outData