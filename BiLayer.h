/*
 * BiLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 *  
 *  Revise on : 2018/2/2
 *      Author: yzhu,suda
 */

#ifndef SRC_BiLayer_H_
#define SRC_BiLayer_H_
#include "../mshadow/tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class BiLayer {

public:

  Tensor<xpu, 2, dtype> _WL;
  Tensor<xpu, 2, dtype> _WR;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 2, dtype> _gradWL;
  Tensor<xpu, 2, dtype> _gradWR;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 2, dtype> _eg2WL;
  Tensor<xpu, 2, dtype> _eg2WR;
  Tensor<xpu, 2, dtype> _eg2b;

  Tensor<xpu, 2, dtype> _meanWL;
  Tensor<xpu, 2, dtype> _meanWR;
  Tensor<xpu, 2, dtype> _meanb;
  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp
  int iter;

public:
  BiLayer() {
  }

  inline void initial(int nOSize, int nLISize, int nRISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nLISize + nRISize + 1));
    //dtype bound = 0.01;

    _WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _gradWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _eg2WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _meanWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);

    _WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _gradWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _eg2WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _meanWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _meanb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_WL, -1.0 * bound, 1.0 * bound, seed);
    random(_WR, -1.0 * bound, 1.0 * bound, seed+1);
    random(_b, -1.0 * bound, 1.0 * bound, seed+2);

    iter = 0;
    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {
    static int nOSize, nLISize, nRISize;
    nOSize = WL.size(0);
    nLISize = WL.size(1);
    nRISize = WR.size(1);

    _WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _gradWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _eg2WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _meanWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    Copy(_WL, WL);

    _WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _gradWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _eg2WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _meanWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    Copy(_WR, WR);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _meanb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);
   
    iter = 0;
    _bUseB = bUseB;
    _funcType = funcType;
  }


  inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, int funcType = 0) {
    static int nOSize, nLISize, nRISize;
    nOSize = WL.size(0);
    nLISize = WL.size(1);
    nRISize = WR.size(1);

    _WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _gradWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _eg2WL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    _meanWL = NewTensor<xpu>(Shape2(nOSize, nLISize), d_zero);
    Copy(_WL, WL);

    _WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _gradWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _eg2WR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    _meanWR = NewTensor<xpu>(Shape2(nOSize, nRISize), d_zero);
    Copy(_WR, WR);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _meanb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    iter = 0;
    _bUseB = false;
    _funcType = funcType;
  }

  inline void release() {
    FreeSpace(&_WL);
    FreeSpace(&_gradWL);
    FreeSpace(&_eg2WL);
    FreeSpace(&_meanWL);
    FreeSpace(&_WR);
    FreeSpace(&_gradWR);
    FreeSpace(&_eg2WR);
    FreeSpace(&_meanWR);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
    FreeSpace(&_meanb);
  }

  virtual ~BiLayer() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradWL);
    result += squarenorm(_gradWR);
    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradWL = _gradWL * scale;
    _gradWR = _gradWR * scale;
    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y) {
    y = dot(xl, _WL.T());
    y += dot(xr, _WR.T());
    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }


  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> xl, Tensor<xpu, 3, dtype> xr, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);
    for(int id = 0; id < seq_size; id++){
      y[id] = dot(xl[id], _WL.T());
      y[id] += dot(xr[id], _WR.T());
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

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& xl, const std::vector<Tensor<xpu, 2, dtype> >& xr,
      std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();
    for(int id = 0; id < seq_size; id++){
      y[id] = dot(xl[id], _WL.T());
      y[id] += dot(xr[id], _WR.T());
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

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lxl, Tensor<xpu, 2, dtype> lxr, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);
    if(bclear){
      lxl = 0.0;
      lxr = 0.0;
    }
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);
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
    _gradWL += dot(cly.T(), xl);
    _gradWR += dot(cly.T(), xr);

    //_gradb
    if (_bUseB)
      _gradb += cly;

    //lx
    lxl += dot(cly, _WL);
    lxr += dot(cly, _WR);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> xl, Tensor<xpu, 3, dtype> xr, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly,
      Tensor<xpu, 3, dtype> lxl, Tensor<xpu, 3, dtype> lxr, bool bclear = false) {
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear){
      lxl = 0.0;
      lxr = 0.0;
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
      _gradWL += dot(cly.T(), xl[id]);
      _gradWR += dot(cly.T(), xr[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lxl[id] += dot(cly, _WL);
      lxr[id] += dot(cly, _WR);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &xl, const std::vector<Tensor<xpu, 2, dtype> > &xr,
      const std::vector<Tensor<xpu, 2, dtype> > &y, const std::vector<Tensor<xpu, 2, dtype> > &ly,
      std::vector<Tensor<xpu, 2, dtype> > &lxl, std::vector<Tensor<xpu, 2, dtype> > &lxr, bool bclear = false) {
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear){
      for (int id = 0; id < seq_size; id++) {
        lxl[id] = 0.0;
        lxr[id] = 0.0;
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
      _gradWL += dot(cly.T(), xl[id]);
      _gradWR += dot(cly.T(), xr[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lxl[id] += dot(cly, _WL);
      lxr[id] += dot(cly, _WR);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void randomprint(int num) {
    static int nOSize, nLISize, nRISize;
    nOSize = _WL.size(0);
    nLISize = _WL.size(1);
    nRISize = _WR.size(1);
    int count = 0;
    while (count < num) {
      int idxl = rand() % nOSize;
      int idyl = rand() % nLISize;
      int idxr = rand() % nOSize;
      int idyr = rand() % nRISize;

      std::cout << "_WL[" << idxl << "," << idyl << "]=" << _WL[idxl][idyl] << " ";
      std::cout << "_WR[" << idxr << "," << idyr << "]=" << _WR[idxr][idyr] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradWL = _gradWL + _WL * regularizationWeight;
    _eg2WL = _eg2WL + _gradWL * _gradWL;
    _WL = _WL - _gradWL * adaAlpha / F<nl_sqrt>(_eg2WL + adaEps);

    _gradWR = _gradWR + _WR * regularizationWeight;
    _eg2WR = _eg2WR + _gradWR * _gradWR;
    _WR = _WR - _gradWR * adaAlpha / F<nl_sqrt>(_eg2WR + adaEps);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }
  
  inline void updateAdam(dtype belta1, dtype belta2, dtype regularizationWeight, dtype adaAlpha, dtype adaEps){    
    dtype lr_t = adaAlpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
   
    _gradWL = _gradWL + _WL * regularizationWeight;   
    _meanWL = belta1 * _meanWL + (1 - belta1) * _gradWL;
    _eg2WL = belta2 * _eg2WL + (1 - belta2)* _gradWL * _gradWL;
    _WL = _WL - _meanWL * lr_t/F<nl_sqrt>(_eg2WL + adaEps);

    _gradWR = _gradWR + _WR * regularizationWeight;   
    _meanWR = belta1 * _meanWR + (1 - belta1) * _gradWR;
    _eg2WR = belta2 * _eg2WR + (1 - belta2) * _gradWR * _gradWR;
    _WR = _WR - _meanWR * lr_t/F<nl_sqrt>(_eg2WR + adaEps);

    
    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;    
      _meanb = belta1 * _meanb + (1 - belta1) * _gradb;
      _eg2b = belta2 * _eg2b + (1 - belta2)* _gradb * _gradb;
      _b = _b - _meanb * lr_t/F<nl_sqrt>(_eg2b + adaEps);
    }

    iter++;                           
    clearGrad();
  }


  inline void clearGrad() {
    _gradWL = 0;
    _gradWR = 0;
    if (_bUseB)
      _gradb = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _WL);
    SaveBinary(outf, _WR);
    SaveBinary(outf, _b);
    
    SaveBinary(outf, _gradWL);
    SaveBinary(outf, _gradWR);
    SaveBinary(outf, _gradb);
    
    SaveBinary(outf, _eg2WL);
    SaveBinary(outf, _eg2WR);
    SaveBinary(outf, _eg2b);

    SaveBinary(outf, _meanWL);
    SaveBinary(outf, _meanWR);
    SaveBinary(outf, _meanb);
    
    WriteBinary(outf, _bUseB);
    WriteBinary(outf, _funcType);
    // cout << "Bilayer " << _bUseB << _funcType << endl;
    // cout << "Bilayer value: " << _WR[1][1] << endl;

  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_WL, false);
    LoadBinary(inf, &_WR, false);
    LoadBinary(inf, &_b, false);
    
    LoadBinary(inf, &_gradWL, false);
    LoadBinary(inf, &_gradWR, false);
    LoadBinary(inf, &_gradb, false);
    
    LoadBinary(inf, &_eg2WL, false);
    LoadBinary(inf, &_eg2WR, false);
    LoadBinary(inf, &_eg2b, false);

    LoadBinary(inf, &_meanWL, false);
    LoadBinary(inf, &_meanWR, false);
    LoadBinary(inf, &_meanb, false);
    
    ReadBinary(inf, _bUseB);
    ReadBinary(inf, _funcType);
    // cout << "Bilayer " << _bUseB << _funcType << endl;
    // cout << "Bilayer value: " << _WR[1][1] << endl;
  }
  
};

#endif /* SRC_BiLayer_H_ */
