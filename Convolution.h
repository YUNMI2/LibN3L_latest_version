/*
 * Convolution.h
 * 
 *
 *  Created on: Nov 16, 2017
 *      Author: yzhu
 */

#ifndef SRC_Convolution_H_
#define SRC_convolution_H_
#include "../mshadow/tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class Convolution {

public:

  Tensor<xpu, 3, dtype> _W;
  Tensor<xpu, 3, dtype> _b;

  Tensor<xpu, 3, dtype> _gradW;
  Tensor<xpu, 3, dtype> _gradb;

  Tensor<xpu, 3, dtype> _eg2W;
  Tensor<xpu, 3, dtype> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp, 4:relu, 5:elu

  int _kernel_size_n = 0;
  int _kernel_size_m = 0;
  int _stride_size_m = 0;
  int _stride_size_n = 0;
  int _n_cnn_filters = 0;

public:
  Convolution() {
  }
  

  inline void initial(int n_cnn_filters,int kernel_size_n,int kernel_size_m,int stride_size_n,int stride_size_m,bool bUseB = true,int seed = 0,int funcType = 0 )
  {
    dtype bound = sqrt(6.0 / (2*kernel_size_n*kernel_size_m+1));

    _kernel_size_n = kernel_size_n;
    _kernel_size_m = kernel_size_m;
    _stride_size_n = stride_size_n;
    _stride_size_m = stride_size_m;
    _n_cnn_filters = n_cnn_filters;


    _W = NewTensor<xpu>(Shape3(n_cnn_filters, kernel_size_n,kernel_size_m), d_zero);//卷积核的大小和数量
    _gradW = NewTensor<xpu>(Shape3(n_cnn_filters, kernel_size_n,kernel_size_m), d_zero);
    _eg2W = NewTensor<xpu>(Shape3(n_cnn_filters, kernel_size_n,kernel_size_m), d_zero);
    
    _b = NewTensor<xpu>(Shape3(n_cnn_filters, 1,1), d_zero);//每个卷积核的偏置
    _gradb = NewTensor<xpu>(Shape3(n_cnn_filters, 1,1), d_zero);
    _eg2b = NewTensor<xpu>(Shape3(n_cnn_filters, 1,1), d_zero);
 
            
    random(_W, -1.0 * bound, 1.0 * bound, seed);
 
    random(_b, -1.0 * bound, 1.0 * bound, seed + 1);

     
    _bUseB = bUseB;
    _funcType = funcType;
  
  }




















/*  inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W, -1.0 * bound, 1.0 * bound, seed);
    random(_b, -1.0 * bound, 1.0 * bound, seed + 1);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W,  int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);


    _bUseB = false;
    _funcType = funcType;
  }

*/
  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }

  virtual ~Convolution() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW);

    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW = _gradW * scale;
    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:




  inline void ComputeForwardScore(Tensor<xpu,2,dtype> x,Tensor<xpu,3,dtype> y)
  {

  //  cout << "Conv::ComFor::x.size(0):" << x.size(0) <<endl;
  //  cout << "Conv::ComFor::y.size(1):" << y.size(1) << endl;
  //  cout << "Conv::ComFor::x.size(1):" << x.size(1) <<endl;
  //  cout << "Conv::ComFor::y.size(2):" << y.size(2) << endl;
    assert(_n_cnn_filters == y.size(0));
    assert(x.size(0) == y.size(1));
    assert(x.size(1) == y.size(2));


    y = 0.0;
    Tensor<xpu, 2, dtype> padding_zero;
    
    padding_zero = NewTensor<xpu>(Shape2(x.size(0)*_stride_size_n+_kernel_size_n-_stride_size_n,x.size(1)*_stride_size_m+_kernel_size_m-_stride_size_m),d_zero);
//    cout << "padding_zero.size(0):: " << padding_zero.size(0) << endl;
//    cout << "padding_zero.size(1):: " << padding_zero.size(1) << endl;





  //  int count = 0;
    for (int idx = 0;idx < x.size(0);idx++)
    {
      for (int idy = 0;idy < x.size(1);idy++)
      {
	//cout << "padding zero" << count << endl;
        padding_zero[idx][idy]+=x[idx][idy];
//	count ++;
      }
    }


//  cout << "hahhah " << endl;
  int cur_idx = 0;
  int cur_idy = 0;




  for (int n_filter = 0;n_filter < _n_cnn_filters;n_filter++)
  {//这边是表示卷积核的数量
    cur_idx = 0;
    for (int idx = 0;(idx < padding_zero.size(0))&&(cur_idx < x.size(0));idx = idx + _stride_size_n,cur_idx++ )
    {
      cur_idy = 0;
    //  cout << idx;
      for (int idy = 0;(idy < padding_zero.size(1))&&(cur_idy < x.size(1));idy = idy + _stride_size_m,cur_idy++)
      {
	 for(int ker_x = 0;ker_x < _kernel_size_n;ker_x++)
	 {
	   for (int ker_y = 0;ker_y < _kernel_size_m;ker_y++)
	   {
	     y[n_filter][cur_idx][cur_idy] = y[n_filter][cur_idx][cur_idy]+ _W[n_filter][ker_x][ker_y]*padding_zero[idx+ker_x][idy+ker_y];	     
	   }
	 }  
         if (_bUseB)
           y[n_filter][cur_idx][cur_idy] = y[n_filter][cur_idx][cur_idy] + _b[n_filter][0][0];
      }
    } 

    //cout << "hahhahah" << endl;
 
    if (_funcType == 0)
      y[n_filter] = F<nl_tanh>(y[n_filter]);
    else if (_funcType == 1)
      y[n_filter] = F<nl_sigmoid>(y[n_filter]);
    else if (_funcType == 3)
      y[n_filter] = F<nl_exp>(y[n_filter]);
    else if (_funcType == 4)
      y[n_filter] = F<nl_relu>(y[n_filter]);
    else if (_funcType == 5)
      y[n_filter] = F<nl_elu>(y[n_filter]);
  } 
 // cout << "hahhahahah" << endl;
  FreeSpace(&padding_zero);



}







 /* inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y) {
    y = dot(x, _W.T());
    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _W.T());
      if (_bUseB)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x, std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _W.T());
      if (_bUseB)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }
*/



  
inline void ComputeBackwardLoss(Tensor<xpu ,2, dtype>x, Tensor<xpu,3,dtype> y,Tensor<xpu,3,dtype>ly,Tensor<xpu,2,dtype>lx,bool bclear = false){
  Tensor<xpu, 3, dtype> deri_yx(Shape3(y.size(0), y.size(1),y.size(2))), cly(Shape3(y.size(0), y.size(1),y.size(2)));
  AllocSpace(&deri_yx);
  AllocSpace(&cly);

  Tensor<xpu, 2, dtype> padding_zero;

  padding_zero = NewTensor<xpu>(Shape2(x.size(0)*_stride_size_n+_kernel_size_n-_stride_size_n,x.size(1)*_stride_size_m+_kernel_size_m-_stride_size_m),d_zero);
    
  for (int idx = 0;idx < x.size(0);idx++)
  {
    for (int idy = 0;idy < x.size(1);idy++)
    {
      padding_zero[idx][idy]+=x[idx][idy];
    }
  }
      
  if(bclear)
    lx = 0.0;
  
 //or (int )
   if (_funcType == 0){
     deri_yx = F<nl_dtanh>(y);
     cly = ly * deri_yx;  
   }else if(_funcType == 1){
     deri_yx = F<nl_dsigmoid>(y);
     cly = ly * deri_yx;
   }else if(_funcType == 3){
     cly = ly * y;
   }else if(_funcType == 4){
     deri_yx = F<nl_drelu>(y);
     cly = ly * deri_yx;
   }else if(_funcType == 5){
     deri_yx = F<nl_delu>(y);
     cly = ly * deri_yx;
   }else{
     Copy(cly,ly);
   }

  int cur_idx = 0;
  int cur_idy = 0;


  assert(_n_cnn_filters == y.size(0));
  for (int n_filters = 0;n_filters < y.size(0);n_filters++ ){
    cur_idx = 0;
    for(int idx = 0 ;(idx < padding_zero.size(0))&&(cur_idx < x.size(0));idx = idx + _stride_size_n,cur_idx++){
      cur_idy = 0;
      for (int idy = 0;(idy < padding_zero.size(1))&&(cur_idy < x.size(1));idy = idy+ _stride_size_m,cur_idy++ ){
        for (int ker_x = 0 ;ker_x < _kernel_size_n;ker_x++){
          for (int ker_y = 0;ker_y < _kernel_size_m;ker_y++){
	    _gradW[n_filters][ker_x][ker_y] += cly[n_filters][cur_idx][cur_idy]*padding_zero[idx+ker_x][idy+ker_y];
	  }	
        }
        _gradb[n_filters][0][0]+= cly[n_filters][cur_idx][cur_idy];
      }
      }
    } 
  FreeSpace(&padding_zero);  
  FreeSpace(&deri_yx);
  FreeSpace(&cly);


}










/*

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly, Tensor<xpu, 2, dtype> lx, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);//导数的英文deri
      cly = ly * deri_yx;
    } else if (_funcType == 1) {
      deri_yx = F<nl_dsigmoid>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 3) {
      cly = ly * y;
    } else {
      //cly = ly;
      Copy(cly, ly);
    }
    //_gradW
    _gradW += dot(cly.T(), x);//(T 我猜是转置)

    //_gradb
    if (_bUseB)
      _gradb += cly;

    //lx
    lx += dot(cly, _W);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _gradW += dot(cly.T(), x[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx[id] += dot(cly, _W);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x, const std::vector<Tensor<xpu, 2, dtype> > &y,
      const std::vector<Tensor<xpu, 2, dtype> > &ly, std::vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      for (int id = 0; id < seq_size; id++) {
        lx[id] = 0.0;
      }
    }
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _gradW += dot(cly.T(), x[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx[id] += dot(cly, _W);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }
*/

  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _W.size(0);
    nISize = _W.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_W[" << idx << "," << idy << "]=" << _W[idx][idy] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW * _gradW;
    _W = _W - _gradW * adaAlpha / F<nl_sqrt>(_eg2W + adaEps);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _gradW = 0;
    if (_bUseB)
      _gradb = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _b);
    SaveBinary(outf, _gradW);
    SaveBinary(outf, _gradb);
    SaveBinary(outf, _eg2W);
    SaveBinary(outf, _eg2b);
    WriteBinary(outf, _bUseB);
    WriteBinary(outf, _funcType);
    // cout << "Convolution " << _bUseB << _funcType << endl;

  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_b, false);
    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_gradb, false);
    LoadBinary(inf, &_eg2W, false);
    LoadBinary(inf, &_eg2b, false);
    ReadBinary(inf, _bUseB);
    ReadBinary(inf, _funcType);
    // cout << "Convolution " << _bUseB << _funcType << endl;
  }
  
};

#endif /* SRC_Convolution_H_ */
