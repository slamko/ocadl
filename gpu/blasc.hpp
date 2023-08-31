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


#endif
