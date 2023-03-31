#ifndef _TEST_H_
#define _TEST_H_
#include <cstdlib>
#include <cmath>

class test_utils {
public:
  // A: M x K; B: K x N; C: M x N;
  static void gemm_ref(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        // C[i,j] = SUM(A[i,k] * B[k,j])
        float sum = ACC ? C[i * ldc + j] : 0;
        for (int k = 0; k < K; ++k) {
          sum += A[i * lda + k] * B[k * ldb + j];   
        }
        C[i * ldc + j] = sum;
      }
    }
  }

  static void gemm_ref_int8(const float *A, const int8_t *B, bool trans_b, float *C, int M, int N, int K, bool ACC) {
    int lda = K, ldc = N, ldb = trans_b?K:N;
    if(!trans_b){
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          // C[i,j] = SUM(A[i,k] * B[k,j])
          float sum = ACC ? C[i * ldc + j] : 0;
          for (int k = 0; k < K; ++k) {
            sum += A[i * lda + k] * (float)B[k * ldb + j];   
          }
          C[i * ldc + j] = sum;
        }
      }
    }
    else{
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          // C[i,j] = SUM(A[i,k] * B[k,j])
          float sum = ACC ? C[i * ldc + j] : 0;
          for (int k = 0; k < K; ++k) {
            sum += A[i * lda + k] * (float)B[j * ldb + k];   
          }
          C[i * ldc + j] = sum;
        }
      }      
    }

  }

  static void add_bias(float *C, float *bias, int M, int N, int ldc) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += bias[j];
      }
    }
  }

  // Transpose B into transB
  // B: shape is K x N
  // transB: shape is N x K
  static void transpose(float *B, float *transB, int K, int N) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        // transB[i, j] = B[j, i];
        transB[i * K + j] = B[j * N + i];
      }
    }
  }
  
  static bool is_same_matrix(const float *C1, const float *C2, int M, int N, int ldc, float threshold) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(C1[i * ldc + j] - C2[i * ldc + j]) > threshold) {
          return false;
        }
      }
    }
    return true;
  }

  static int diff_index(const float *C1, const float *C2, int M, int N, int ldc, float threshold) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(C1[i * ldc + j] - C2[i * ldc + j]) > threshold) {
          return i * ldc + j;
        }
      }
    }
    return -1;
  }

  static void init(float *buf, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand() / RAND_MAX;
        //buf[i] = 1.0f;
    }
  }

  static void init_int8(int8_t *buf, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand();
        //buf[i] = 1.0f;
    }
  }

};
#endif