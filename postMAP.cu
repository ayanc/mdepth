/* 
--Ayan Chakrabarti <ayanc@ttic.edu>
*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdint.h>

#define F float

#define NUMT 1024

void __global__ postMAP(F * der, F * pred, F * bins, F beta,
			int W, int H, int K, int B, int crop) {


  int i,j,x,y,k,W2,H2;
  F btp1,brat, dmin, cmin, dj, cj, dcur;

  btp1 = 1.0 + beta; brat = beta / btp1;
  W2 = W + 2*crop;
  H2 = H + 2*crop;
  
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < W*H*K;
       i += blockDim.x * gridDim.x) {

    k = i/(W*H); x = i%(W*H); y = x%H; x = x/H;

    cmin = pred[y+x*H+k*W*H];
    dmin = bins[k]; 

    if(beta > 0.0) {
      dcur = der[(y+crop)+(x+crop)*H2+k*W2*H2];
      cmin = cmin + brat*(dmin-dcur)*(dmin-dcur);
      dmin = (dmin + beta*dcur) / btp1;
    }

    for(j = 1; j < B; j++) {
      cj = pred[y+x*H+k*W*H+j*W*H*K];
      dj = bins[k+j*K];

      if(beta > 0.0) {
	cj = cj + brat*(dj-dcur)*(dj-dcur);
	dj = (dj + beta*dcur) / btp1;
      }

      if(cj < cmin) {cmin = cj; dmin = dj;};
    }

    der[(y+crop)+(x+crop)*H2+k*W2*H2] = dmin;
  }

}


F * getGPUmem(const char * name) {

  const mxGPUArray * tmp;
  F * dptr;

  if(!mxIsGPUArray(mexGetVariablePtr("caller",name)))
    mexPrintf("%s is not on gpu!\n",name);

  tmp = mxGPUCreateFromMxArray(mexGetVariablePtr("caller",name));
  dptr = (F*) mxGPUGetDataReadOnly(tmp);
  mxGPUDestroyGPUArray(tmp);

  return (F*) dptr;
}

/*
  function postMAP(beta,crop,H,W,K,B)
       X, pred, bins need to be present in the caller workspace.
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  F * der, * pred, * bins, beta;
  int crop,H,W,K,B;

  beta = mxGetScalar(prhs[0]);
  crop = (int) mxGetScalar(prhs[1]);
  H = (int) mxGetScalar(prhs[2]);
  W = (int) mxGetScalar(prhs[3]);
  K = (int) mxGetScalar(prhs[4]);
  B = (int) mxGetScalar(prhs[5]);

  
  der = getGPUmem("X"); pred = getGPUmem("pred"); bins = getGPUmem("bins");
  postMAP<<<(W*H*K+NUMT-1)/NUMT,NUMT>>>(der,pred,bins,beta,W,H,K,B,crop);
}
