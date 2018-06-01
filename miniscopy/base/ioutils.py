#!/usr/bin/env python3

import os, sys, re, glob

"""
Copyright (C) 2018 Denis Polygalov,
Laboratory for Circuit and Behavioral Physiology,
RIKEN Center for Brain Science, Saitama, Japan.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, a copy is available at
http://www.fsf.org/
"""

def enum_video_files_dir(target_source, s_wildcard):
    """
    See enum_video_files() for details.
    """
    # target_source = "some_dir8"
    # s_wildcard = "msCam*.avi" or "behavCam*.avi"
    l_file_names = glob.glob(os.path.join(target_source, s_wildcard))
    # l_file_names = ['some_dir8\msCam1.avi', 'some_dir8\msCam10.avi', 'some_dir8\mcCam2.avi', ...]
    l_file_numbers = []
    # l_file_numbers = [1, 10, 2, ...]
    
    if len(l_file_names) == 0:
        raise IOError("No matching files found")
    #
    for s_fname in l_file_names:
        if not os.path.isfile(s_fname):
            raise IOError("Not a regular file: %s" % s_fname)
        #
        if not os.access(s_fname, os.R_OK):
            raise IOError("Access denied for file: %s" % s_fname)
        #
        # os.path.split(s_fname)[-1] will be only the file name, i.e. 'mcCam2.avi' etc.
        l_numbers_in_fname = re.findall(r'\d+', os.path.split(s_fname)[-1])
        if len(l_numbers_in_fname) != 1:
            raise ValueError("Wrong file name format: %s" % s_fname)
        try:
            l_file_numbers.append(int(l_numbers_in_fname[0]))
        except ValueError:
            raise ValueError("Wrong file number: %s" % s_fname)
        #
    # sort numbers in file names
    l_idx = [i[0] for i in sorted(enumerate(l_file_numbers), key=lambda x:x[1])]
    l_sorted_file_names   = [l_file_names[i]   for i in l_idx]
    l_sorted_file_numbers = [l_file_numbers[i] for i in l_idx]
    
    # calculate difference between adjacent file numbers and check for missing files
    if len(l_sorted_file_numbers) >= 2:
        l_fnum_diff = [j - i for i, j in zip(l_sorted_file_numbers[:-1], l_sorted_file_numbers[1:])]
        if sum(l_fnum_diff) != len(l_fnum_diff):
            raise ValueError("Missing files (holes in numbering) found!")
        #
    #
    # another method
    if max(l_sorted_file_numbers) != len(l_sorted_file_names):
        raise ValueError("Missing files (length mismatch) found!")
    #
    if __debug__:
        for i in range(len(l_file_names)):
            print("DEBUG: %s\t%d\t%s" % (l_file_names[i], l_file_numbers[i], l_sorted_file_names[i]))
        #
    #
    return tuple(l_sorted_file_names)
#

def enum_video_files_txt(target_source):
    """
    See enum_video_files() for details.
    """
    if not os.access(target_source, os.R_OK):
        raise IOError("Access denied for file: %s" % target_source)
    #
    l_out_file_names = []
    with open(target_source, 'r') as f:
        for s_fname in f:
            s_fname = s_fname.strip()
            if len(s_fname) == 0: continue
            if s_fname.startswith('#'): continue
            if not os.path.isfile(s_fname): raise IOError("Not a regular file: %s" % s_fname)
            if not os.access(s_fname, os.R_OK): raise IOError("Access denied for file: %s" % s_fname)
            l_out_file_names.append(s_fname)
        #
    #
    if len(l_out_file_names) == 0:
        raise IOError("No input files found")
    #
    return tuple(l_out_file_names)
#

def enum_video_files(target_source, s_wildcard):
    """
    Return a tuple of strings pointed to a set of input *.avi files.
    The 'target_source' can be a path to a directory containing the set,
    or a file name of a text file. The text file then contain a list of
    paths to avi files.
    Example:
    >>> t_files = enum_video_files("H14_M31_S15", "msCam*.avi")
    >>> t_files = enum_video_files("subject_12/4_6_2017/H14_M31_S15", "msCam*.avi")
    >>> t_files = enum_video_files("H14_M31_S15", "behavCam*.avi")
    >>> t_files = enum_video_files("file_list.txt", None)
    Note that no additional sorting will be applied in the case of 
    loading from a text file. Files names will be loaded as is.
    """
    if os.path.isdir(target_source):
        return enum_video_files_dir(target_source, s_wildcard)
    #
    elif os.path.isfile(target_source):
        return enum_video_files_txt(target_source)
    #
    else:
        raise ValueError("Unknown target. It must be an existed directory or file name")
    #
#

