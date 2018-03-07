# Written to profile gamma
# Suryarao Bethapudi


n_iter = 10
twos = range(5,20)

# No edits beyond this line for profiling
##################################################
import sys

if len(sys.argv) < 2:
    print "Gamma profiler"
    print "-----------------------------"
    print "<program> <mode> <filename>"
    print " 'profile' --> Does timing "
    print " 'plot'    --> Does plotting "
    print " put logs into <filename> "
    sys.exit(0)

mode = sys.argv[1].lower()
fh = open(sys.argv[2],"r+")
fl = open("log","r+")
cc = "FFTCHIRP, DDTCHIRP, FFT, DDT, MSE\n"
fl.write(cc)
fl.close()

if mode == 'profile' :
    # profile logic
    for i in range(n_iter):
        for j in twos:
            cc = "gamma 42.42 " + str(j) + " eventhis >> log\n"
            fh.write(cc)
    
elif mode == 'plot':
    # plot logic
    import numpy as np
    import matplotlib.pyplot as plt
    #
    fig, ax = plt.subplots(1,2)
    # TODO

else:
    print "Unrecognized Mode"
    sys.exit(1)

# save and exit
fh.close()
