import visvis as vv
import numpy as np


from matplotlib.image import imread
from matplotlib.cbook import get_sample_data

app = vv.use()
imgdata = imread(get_sample_data('test0.png'))
for i in range(144):
    globals()['imgdata%s' % i] = imread(get_sample_data('test'+str(i)+'.png'))

nr, nc = imgdata.shape[:2]
x,y = np.mgrid[:nr, :nc]
z = np.ones((nr, nc))

for s in range(10):
    vv.surf(x, y, z*s*50, globals()['imgdata%s' % s])


app.Run()