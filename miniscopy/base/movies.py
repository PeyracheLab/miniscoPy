import numpy as np
import cv2

class Movie(object):
	"""
		To play the movie
		To not use for big movie as it may cause memory error
		the best is to load in memory a chunk of the movie and instantiate a movie object
	"""

	def __init__(self, data):
		self.T, self.H, self.W = data.shape
		self.data = data

	def play(self, gain = 1, magnification = 1, looping = True, fr = 30):
		maxmov = np.nanmax(self.data)
		end = False		
		while looping:
			for i in range(self.T):
				frame = self.data[i]
				if magnification != 1:
					frame = cv2.resize(frame, None, fx = magnification, fy = magnification, interpolation = cv2.INTER_LINEAR)
				cv2.imshow('frame', frame * gain / maxmov)
				if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
					looping = False
					end = True
					break
			if end:
				break
		cv2.waitKey(100)
		cv2.destroyAllWindows()
		for i in range(10):
			cv2.waitKey(100)

		return