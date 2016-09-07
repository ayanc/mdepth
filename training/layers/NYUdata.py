# Data layer for loading NYUv2 images.
# See provided train.prototxt for usage.
#-- Ayan Chakrabarti <ayanc@ttic.edu>

import caffe
import numpy as np

from os import getenv
from skimage.io import imread
from scipy.ndimage import zoom, rotate



ht_=427
wd_=561

ht2_=12*32
wd2_=16*32

def rand_aug(im,z):
    
    ang = np.float32(np.random.random(1)*10.0-5.0); ang = ang[0]
    rgb = np.float32(np.random.random(3)*0.4+0.8);
    ctrst = np.float32(np.random.random(1)+0.5); ctrst = ctrst[0]

    im = rotate(im,ang,reshape=False,mode='nearest',order=1).copy()
    z = rotate(z,ang,reshape=False,mode='nearest',order=1).copy()

      
    for j in range(3):
        im[:,:,j] = im[:,:,j] * rgb[j]

    im[...] = np.minimum(255.0,(im[...] ** ctrst) * (255.0**(1.0-ctrst)))
    return im,z

class data(caffe.Layer):
    
    def setup(self,bottom,top):
        sdir = getenv('NYU_DATA_DIR')
        if sdir is None:
            raise Exception("Set environment variable NYU_DATA_DIR "
                            " to data directory.")

        params = self.param_str.split(':')
        if len(params) != 5:
            raise Exception("NYUdata params must be of the form "
                            "listfile:batch_size:receptive_size:"
                            "filter_size:iftest")

        # Read list of files
        self.flist = [sdir + "/" + line.rstrip('\n') for
                      line in open(params[0])]

        # Read remaining params
        self.bsize = int(params[1])
        self.rsize = int(params[2])
        self.fsize = int(params[3])

        self.zx = int((self.fsize-1)/2)
        self.zy = int((self.fsize-1)/2-5)

        self.ix = int(self.rsize)
        self.iy = int(self.rsize-5)

        self.iyp = int(np.maximum(self.iy,0))
        self.ixp = int(np.maximum(self.ix,0))

        self.test = params[4] == '1'

        # Shuffle file list
        np.random.shuffle(self.flist) if not self.test else None

        # Setup walk through database
        self.imid = 0

    def reshape(self,bottom,top):
        top[0].reshape(self.bsize,3,ht_+2*self.iy,wd_+2*self.ix)
        top[1].reshape(self.bsize,1,ht_+2*self.zy,wd_+2*self.zx)
        top[2].reshape(self.bsize,1,ht_+2*self.zy,wd_+2*self.zx)
        top[3].reshape(self.bsize,3,ht2_,wd2_)
        
    def forward(self,bottom,top):
        for i in range(self.bsize):
            base=self.flist[self.imid]

            im = imread(base+"_i.png")
            z = imread(base+"_f.png")
            im = np.float32(im)
            z = np.float32(z)
            
            if not self.test:
                im,z = rand_aug(im,z)

                ref = np.random.randint(0,2)
                if ref == 1:
                    im = im[:,::-1,:].copy();
                    z = z[:,::-1].copy();

            # Cropped image for VGG path
            imVGG = im[21:-22,24:-25,:].copy();
            top[3].data[i,0,:,:] = (imVGG[:,:,2]-103.939)
            top[3].data[i,1,:,:] = (imVGG[:,:,1]-116.779)
            top[3].data[i,2,:,:] = (imVGG[:,:,0]-123.68)
            
            # Do the cropping / padding

            # We'll assume Z always needs padding not cropping
            z = np.pad(z,((self.zy,self.zy),(self.zx,self.zx)),'edge').copy()
            # Image might need cropping in y
            impad = ((self.iyp,self.iyp),(self.ixp,self.ixp),(0,0))
            im = np.pad(im,impad,'edge').copy()
            if self.iy < 0:
                im = im[-self.iy:self.iy,:,:].copy()

            msk = np.float32(z > 1)

            # Normalize depth to 0 to 10, and image to [-1,1]
            z = z / (2.0**16.0 - 1.0) * 10.0
            im = 2.0*im/255.0 - 1.0
            
            for j in range(3):
                top[0].data[i,j,...] = im[:,:,j]
            top[1].data[i,...] = z
            top[2].data[i,...] = msk

            
            self.imid=self.imid+1
            if self.imid >= len(self.flist):
                self.imid=0
                np.random.shuffle(self.flist) if not self.test else None
                

    def backward(self,top,propagate_down,bottom):
        pass
