#include "test.h"
#include "timer.h"
#include "gemm_kernel.h"
// const int BLOCK_M = 4, BLOCK_N = 64, BLOCK_K = 256;



void test_gemm(int M, int N, int K, bool trans_b) {
    int lda = K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    float *A = (float *)aligned_alloc(64, M * K * sizeof(float));
    int8_t *B = (int8_t *)aligned_alloc(64, K * N * sizeof(int8_t));
    int8_t *B_pack = (int8_t*)aligned_alloc(64, K * N * sizeof(int8_t));
    float *refC = (float *)aligned_alloc(64, M * N * sizeof(float));
    float *bias = (float *)aligned_alloc(64, N * sizeof(float));
    zero_fill(refC, M, N, ldc);

    test_utils::init(A, M * lda);
    test_utils::init_int8(B, K * N);
    //zero_fill(bias, 1, N, N);
    test_utils::init(bias, N);


    //calculate the reference answer
    test_utils::gemm_ref(A, B, trans_b, refC, M, N, K, false);
    test_utils::add_bias(refC, bias, M, N, ldc);

    pack(B, B_pack, K, N, ldb, trans_b);  



    //test per channel
    int32_t* zero_point_per_channel = (int32_t *)aligned_alloc(64, N * sizeof(float));
    float* scale_per_channel = (float *)aligned_alloc(64, N * sizeof(float));  
    for(int i = 0 ; i < N; i++){
        zero_point_per_channel[i] = 0;
        scale_per_channel[i] = 1.0;
    } 

    float *C_per_channel = (float *)aligned_alloc(64, M * ldc * sizeof(float)); 

    woq_gemm(A, B_pack, C_per_channel, M, N, K, lda, N, ldc, zero_point_per_channel, scale_per_channel, bias);

    if (!test_utils::is_same_matrix(refC, C_per_channel, M, N, ldc, 0.0001f)) {
        int idx = test_utils::diff_index(refC, C_per_channel, M, N, ldc, 0.0001f);
        printf("\tquant per channel Failed: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, trans_b=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
               M, N, K, lda, ldb, ldc, trans_b, idx, refC[idx], idx, C_per_channel[idx]);
    } else {
        printf("\tquant per channel Passed: M=%d, N=%d, K=%d, trans_b=%d\n", M, N, K, trans_b);
    }




    //test per tensor
    int32_t zero_point_per_tensor = 0, scale_per_tensor = 1.0; 

    float *C_per_tensor = (float *)aligned_alloc(64, M * ldc * sizeof(float)); 

    woq_gemm(A, B_pack, C_per_tensor, M, N, K, lda, N, ldc, zero_point_per_tensor, scale_per_tensor, bias);

    if (!test_utils::is_same_matrix(refC, C_per_tensor, M, N, ldc, 0.0001f)) {
        int idx = test_utils::diff_index(refC, C_per_tensor, M, N, ldc, 0.0001f);
        printf("\tquant per tensor Failed: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, trans_b=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
               M, N, K, lda, ldb, ldc, trans_b, idx, refC[idx], idx, C_per_tensor[idx]);
    } else {
        printf("\tquant per tensor Passed: M=%d, N=%d, K=%d, trans_b=%d\n", M, N, K, trans_b);
    }      




    free(A);
    free(B);
    free(B_pack);
    free(C_per_channel);
    free(C_per_tensor);
    free(refC);
    free(zero_point_per_channel);
    free(scale_per_channel);
}


int main() {
    int mnk[][3] = {
        {7, 4095, 4095},
        {7, 16383, 4095},
        {7, 4095,16383},
        {4, 4096, 4096},
        {4, 4096, 16384},
        {4, 16384, 4096},
    };


    for (int i = 0; i < sizeof(mnk) / sizeof(mnk[0]); ++i) {
        test_gemm(mnk[i][0], mnk[i][1], mnk[i][2], true);
        test_gemm(mnk[i][0], mnk[i][1], mnk[i][2], false);
        //benchmark_libxsmm(mnk[i][0], mnk[i][1], mnk[i][2]);
    }

    return 0;
}
