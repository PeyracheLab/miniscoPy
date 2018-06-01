#!/usr/bin/env python3

import os, sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # optionally:
    # os.sys.path.append("D:\\ ... path to \\miniscoPy\\")
    # or:
    # os.sys.path.append("/home/user/ ... path to /miniscoPy/")
    
    from miniscopy.base.ioutils import enum_video_files
    from miniscopy.base.mupamovie import CMuPaMovieCV
    
    if len(sys.argv) < 2:
        print("ERROR: not enough input arguments. Usage:")
        print("%s folder_name_containing_video_files" % sys.argv[0])
        sys.exit(0)
    #
    print("\n")
    t_files = enum_video_files(sys.argv[1], "msCam*.avi")
    oc_movie = CMuPaMovieCV(t_files)
    max_nframes = oc_movie.na_ends[-1]
    
    img = None
    brake = 0
    while(oc_movie.read_next_frame()):
        print("Current abs. frame: %d\t file number: %d\t rel. frame: %d" % \
             (oc_movie.i_curr_abs_frame_num, \
              oc_movie.i_curr_file_idx, \
              oc_movie.i_curr_rel_frame_num) )
        # do something with oc_movie.na_frame
        # ...
        # or just plot it:
        if img is None:
            img = plt.imshow(oc_movie.na_frame)
        else:
            img.set_data(oc_movie.na_frame)
            # alternative:
            # img.set_data(oc_movie.na_frame.astype(np.float32))
        # Wait for any key. Exit if pressed.
        if plt.waitforbuttonpress(timeout=0.05): break
        plt.draw()
        
        brake += 1
        if brake >= 1010: break
    #
#
