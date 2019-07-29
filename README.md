# FourierAnimation
Simple Python package script for animating the reconstruction of any contour with the sum of a complex Fourier series. Inspired by [3Blue1Brown's video](https://www.youtube.com/watch?v=r6sGWTCMz2k) on the topic.

## Required Packages
- Numpy
- OpenCV
- Scipy

## Example Usage
```
import FourierAnimation as fa
import cv2

contour = <your n x 2 array of ordered (x, y) contour points>
imshape = <(h x w) size of output image>
c, nlist = fa.calculateFourierSeries(contour, nvec=100) #calculate Fourier coefficients
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('fourier.avi', fourcc, 30, (imshape[1], imshape[0])) #video writer
fa.animate(c, nlist, imshape, save=True, vidwriter=out)
```
## Demo
Watch [here](https://youtu.be/rcA26UL32Pc) for demo performance on complex and simple contours.

## License
[MIT](https://choosealicense.com/licenses/mit/)
