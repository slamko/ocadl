#ifndef BLAC_HPP
#define BLAC_HPP

extern "C" {
#include "blasc.h"
}

class Matrix {
  
public:
  mat matrix;
  
  Matrix(mat mat) : matrix(mat) {} ;

  void print(); 

  ~Matrix();
};


bool mat_cmp(const Matrix &, const Matrix &);

int convolve(const struct mat *input,
             const struct mat *kernels,
             unsigned long padding,
             unsigned long res_width,
             unsigned long res_height,
             struct mat *res);

#endif
