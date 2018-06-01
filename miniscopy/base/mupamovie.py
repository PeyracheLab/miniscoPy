#!/usr/bin/env python3

import numpy as np
import pandas
import cv2

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

class CMuPaMovie(object):
    """
    Base class represents a MultiPart Movie.
    The MultiPart Movie constructed from a tuple of video file names.
    The file names in the tuple must be in correct temporal order.
    """
    def __init__(self, t_file_names):
        self.t_file_names = t_file_names
        self.df_info = pandas.DataFrame( \
            index = np.arange(len(self.t_file_names)), \
            columns = ['file_name', 'start', 'end', 'duration', 'frames', 'frame_rate', 'width', 'height', 'format'] \
        )
        self.t_vid_files   = None
        self.t_vid_streams = None
        self.na_ends  = None # [1000, 2000, 3000, 4000, 4963], read only!
        self.na_frame = None # the frame (as Numpy array)
        self.i_curr_file_idx  = 0     # read only!
        self.i_curr_rel_frame_num = 0 # read only!
        self.i_curr_abs_frame_num = 0 # read only!
        self.i_next_abs_frame_num = 0 # read only!
    #
    def abs2rel(self, i_abs):
        # if requested i_abs is before the start - return (first_file, first_frame)
        if i_abs < 0: return (0,0)
        # if requested i_abs is after the end - return (last_file, last_frame)
        if i_abs >= self.na_ends[-1]: return ( self.na_ends.shape[0] - 1, self.na_ends[-1] - 1 )
        # find an index of the first edge bigger than requested i_abs
        i_1st_bigger_edge = np.argmax(self.na_ends > i_abs)
        # if requested i_abs is within the first bin - return it as is
        if i_1st_bigger_edge == 0:
            return (0, i_abs)
        else:
            return (i_1st_bigger_edge, i_abs - self.na_ends[i_1st_bigger_edge - 1])
        #
    #
    def rel2abs(self, file_idx, frame_num):
        if file_idx < 0: return 0
        if file_idx == 0: return frame_num
        if file_idx >= self.na_ends.shape[0]: return self.na_ends[-1] - 1
        if frame_num >= self.na_ends[file_idx]:
            raise ValueError("Wrong input: %s" % repr((file_idx, frame_num)) )
        #
        # i_abs, absolute frame number 0 ~ self.na_ends[-1]
        return self.na_ends[file_idx - 1] + frame_num
    #
#

class CMuPaMovieCV(CMuPaMovie):
    """
    Class represents a MultiPart Movie (OpenCV-based backend).
    The MultiPart Movie constructed from a tuple of video file names.
    The file names in the tuple must be in correct temporal order.
    For details refer to documentation for the base class CMuPaMovie()
    """
    def __init__(self, t_file_names):
        super().__init__(t_file_names)
        
        # to be turned into tuples at the end of this constructor
        l_vid_files = []
        l_vid_streams = []
        
        for idx in range(len(self.t_file_names)):
            self.df_info.loc[idx, 'file_name'] = self.t_file_names[idx]
            
            hCap = cv2.VideoCapture(self.t_file_names[idx])
            while not hCap.isOpened():
                hCap = cv2.VideoCapture(self.t_file_names[idx])
                cv2.waitKey(1000)
                print("WARNING: waiting for the cv2.VideoCapture()...")
            #
            self.df_info.loc[idx, 'duration'] = hCap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.df_info.loc[idx, 'frames']   = int(hCap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.df_info.loc[idx, 'frame_rate'] = float(hCap.get(cv2.CAP_PROP_FPS))
            self.df_info.loc[idx, 'width']  = int(hCap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.df_info.loc[idx, 'height'] = int(hCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # self.df_info.loc[idx, 'format'] = hCap.get(cv2.CAP_PROP_ ... )
            
            l_vid_files.append( hCap )
            l_vid_streams.append( hCap )
        #
        
        # check if all video files have the same frame width and height
        if self.df_info['width'].sum() != \
           self.df_info['width'][0] * len(self.df_info['width']):
            raise ValueError("Frame width is not consistent across input video files")
        if self.df_info['height'].sum() != \
           self.df_info['height'][0] * len(self.df_info['height']):
            raise ValueError("Frame height is not consistent across input video files")
        #
        
        self.df_info['start'] = self.df_info['duration'].cumsum() - self.df_info['duration']
        self.df_info['end']   = self.df_info['duration'].cumsum()
        # self.df_info = self.df_info.set_index(['file_name'], append = True)
        
        # freeze lists into tuples
        self.t_vid_files = tuple(l_vid_files)
        self.t_vid_streams = tuple(l_vid_streams)
        
        # [1000, 2000, 3000, 4000, 4963], read only!
        self.na_ends = np.array(self.df_info['end'], dtype=np.int32)
        
        if __debug__:
            print(self.df_info)
            print("Total number of frames: %d" % self.na_ends[-1])
        #
    #
    def _read_frame(self, file_idx, frame_num):
        b_ret = False
        if file_idx  >= len(self.t_file_names): return b_ret
        if frame_num >= self.df_info.at[file_idx, 'frames']: return b_ret
        
        # set position to read the requested frame from requested video file
        b_ret = self.t_vid_streams[file_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if not b_ret: return b_ret
        
        # try to read the frame
        while True:
            b_ret, self.na_frame = self.t_vid_streams[file_idx].read()
            if b_ret:
                self.i_curr_file_idx = file_idx
                self.i_curr_rel_frame_num = frame_num
                self.i_curr_abs_frame_num = self.rel2abs(self.i_curr_file_idx, self.i_curr_rel_frame_num)
                return b_ret
            else:
                print("WARNING: waiting for the cv2.read()...")
                cv2.waitKey(1000)
            #
        #
        return b_ret
    #
    def read_frame(self, abs_frame_num):
        rel_file_idx, rel_frame_num = self.abs2rel(abs_frame_num)
        return self._read_frame(rel_file_idx, rel_frame_num)
    #
    def read_next_frame(self):
        rel_file_idx, rel_frame_num = self.abs2rel(self.i_next_abs_frame_num)
        b_ret = self._read_frame(rel_file_idx, rel_frame_num)
        if b_ret == True:
            self.i_next_abs_frame_num += 1
        return b_ret
    #
#
