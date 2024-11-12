import openslide
import numpy as np
import cv2

def get_ndpi_data(path):
	k = 1
	source = openslide.open_slide(path)
	downsamples=source.level_downsamples
	[w,h]=source.level_dimensions[0]
	size1=int(w*(downsamples[0]/downsamples[k]))
	size2=int(h*(downsamples[0]/downsamples[k]))
	region=np.array(source.read_region((0,0),k,(size1,size2)))[:, :, :3]
	return region

path_ndpi = r'ndpi_data/CL-L15-6-D5-2-IL-6.ndpi'
nd_img = get_ndpi_data(path_ndpi)
print("img_shpe:",nd_img.shape)
img_opencv = cv2.cvtColor(nd_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('result_png.jpg', img_opencv, [cv2.IMWRITE_PNG_COMPRESSION, 5])
#print('img shape:',img_opencv.shape)