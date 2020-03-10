#!/usr/bin/env python

import re
import os
import miricoord.imager.mirim_tools as mt
import miricoord.lrs.lrs_tools as lrst
import miricoord.mrs.mrs_tools as mrst
dist_ver = mrst.version()
print('Distortion solution: ', dist_ver)

#import pysiaf
#siaf = pysiaf.Siaf('MIRI')#,basepath='/Users/dlaw/jwcode/pysiaf/pysiaf/pre_delivery_data/MIRI')

def get_test_file(fnum):
    """
    Relative pointer to current working directory for test files.
    
    Parameters: 

        fnum - int variable (1,2,3) indicating which test input file to use.
    
    Returns:
    
        reffile - String path to selected test input file. 
    """

    rootdir= '.'

    if (fnum == 1):
        file='untitled.pointing'
    elif (fnum == 2):
        file='test.pointing'
    elif (fnum == 3):
        file='sub.pointing'
        
    reffile=os.path.join(rootdir,file)
   
    return reffile


def get_data(file):
    """
    Opens a file, and reads in data.
    
    Parameters:
        
        file - file path
        
    Returns:
        
        data - array of strings
    """
    f = open(file,'r')
    data = f.read()
    f.close()
    
    return data

def ref_mode(mode):
    """
    Defines reference pixels for different imaging modes.
    
    Parameters:
        
        mode - string containing imaging mode.
        
    Returns:
    
        xref, yref - Floating point reference pixel coordinates
    
    """
    
    xref, yref = 692.5, 511.5
    xref_slit, yref_slit = 325.13, 299.7
    xref_slitless, yref_slitless = 37.5, 300.
    
    BRIGHTSKY_x, BRIGHTSKY_y = 711.5, 305.5
    SUB256_x, SUB256_y = 539.5, 177.5
    SUB128_x, SUB128_y =  69.5, 951.5
    SUB64_x, SUB64_y =  37.5, 809.5
        
    if "SLITLESS" in mode:
        xref = xref_slitless
        yref = yref_slitless

    elif "SLIT" in mode:
        xref = xref_slit
        yref = yref_slit

    elif "BRIGHTSKY" in mode:
        xref = BRIGHTSKY_x
        yref = BRIGHTSKY_y

    elif "256" in mode:
        xref = SUB256_x
        yref = SUB256_y
        
    elif "128" in mode:
        xref = SUB128_x
        yref = SUB128_y

    elif "64" in mode:
        xref = SUB64_x
        yref = SUB64_y

    else:
        xref = xref
        yref = yref
    
    return xref, yref

def print_head(f):
    """
    Prints currently relevant header information to top of output file.
    
    Parameters:
    
        f - file IO object
    """
    xref, yref = 692.5, 511.5
    xref_slit, yref_slit = 325.13, 299.7
    xref_slitless, yref_slitless = 37.5, 300.
    
    BRIGHTSKY_x, BRIGHTSKY_y = 711.5, 305.5
    SUB256_x, SUB256_y = 539.5, 177.5
    SUB128_x, SUB128_y =  69.5, 951.5
    SUB64_x, SUB64_y =  37.5, 809.5

    f.write('# Dithers are multiplied by -1 pending resolution of http://www.miricle.org/bugzilla/show_bug.cgi?id=588 \n')
    f.write('# The following reference pixels are hard-coded for use: \n')
    f.write('# Imaging: {}, {} \n'.format(xref,yref))
    f.write('# LRS Slit: {}, {} \n'.format(xref_slit,yref_slit))
    f.write('# LRS Slitless: {}, {} \n'.format(xref_slitless,yref_slitless))
    f.write('# BRIGHTSKY: {}, {} \n'.format(BRIGHTSKY_x,BRIGHTSKY_y))
    f.write('# SUB256: {}, {} \n'.format(SUB256_x, SUB256_y))
    f.write('# SUB128: {}, {} \n'.format(SUB128_x, SUB128_y))
    f.write('# SUB64: {}, {} \n\n'.format(SUB64_x, SUB64_y))
    
    return

def make_dith_file(in_file, outfile):
    """
    Converts an APT pointing file to a list of dithers for use in MIRIsim. 
    
    Parameters:
         
        in_file - APT pointing file path
        
    Returns:
    
        outfile - Text file containing MIRIsim readable dithers.

    """
    
    #Read apt data and split into rows
    data = get_data(in_file)
    split_data = data.split('\n')

    #open output file
    f = open(outfile,"w+")
    
    #append header info to output file
    print_head(f)
    
    for row in split_data:
        #split row into columns on whitespace
        r = row.split()

        #rows with < 20 columns contain no data
        if len(r) < 20:
            f.write(str('#' + ' '.join(r)+ '\n'))

        else:
            # 'MIRIM' indicates Imager or LRS
            if "MIRIM" in r[4]:

                v2 = float(r[13])
                v3 = float(r[14])

                #convert v2 and v3 coordinates to x and y
                x,y = mt.v2v3toxy(v2,v3,'F770W')
                
                #determine the proper reference pixel for the imaging mode
                xref, yref = ref_mode(r[4])
                    
                #these keywords all indicate a coronagraphic image, not supported by MIRIsim
                if "BLOCK" in r[4] or "UR" in r[4] or "MASK" in r[4]:
                    f.write("#MIRIsim does not support Coronagraphy \n")
                    continue
                
                #compute dither
                else:
                    dx = xref - x[0]
                    dy = yref - y[0]
                
                #write to file
                try:
                    s = "{0:.2f}, {1:.2f}".format(dx,dy)
                    f.write(s + '\n')
                except:
                    pass

            #'MIRIFU' indicates MRS
            elif "MIRIFU" in r[4]:
                v2 = float(r[13])
                v3 = float(r[14])

                #determine stype
                channel = r[4][-2:]

                #convert to alpha,beta
                a,b = mrst.v2v3toab(v2,v3, channel) #mult by -1
                da = -1.*a
                db = -1.*b

                try:
                    s = "{0:.3f}, {1:.3f}".format(da,db)
                    f.write(s + '\n')
                except:
                    pass

    f.close()
    return

if __name__ == "Main":
    infile = input("APT pointing file name: ")
    outfile = input("Output file name: ")
    
    make_dith_file(infile, outfile)
