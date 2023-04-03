#ifndef _GEMM_KERNEL_H_
#define _GEMM_KERNEL_H_
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <libxsmm.h>
#include <immintrin.h>
#include <emmintrin.h>

const int BLOCK_M = 4, BLOCK_N = 64, BLOCK_K = 256;


struct DotMicroKernelKey {
    bool trans_a;
    bool trans_b;
    int lda;
    int ldb;
    int ldc;

    DotMicroKernelKey(bool trans_a, bool trans_b, int lda, int ldb, int ldc)
        : trans_a(trans_a), trans_b(trans_b), lda(lda), ldb(ldb), ldc(ldc) {}

    bool operator==(const DotMicroKernelKey& other) const {
        return trans_a == other.trans_a && trans_b == other.trans_b && lda == other.lda && ldb == other.ldb && ldc == other.ldc;
    }
};

namespace std {
    template<>
    struct hash<DotMicroKernelKey> {
        std::size_t operator()(const DotMicroKernelKey& key) const {
            std::size_t h = std::hash<bool>()(key.trans_a);
            h = std::hash<bool>()(key.trans_b) ^ (h << 1);
            h = std::hash<int>()(key.lda) ^ (h << 1);
            h = std::hash<int>()(key.ldb) ^ (h << 1);
            h = std::hash<int>()(key.ldc) ^ (h << 1);
            return h;
        }
    };
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
class DotMicroKernel {
public:
    DotMicroKernel(bool trans_a, bool trans_b, int lda, int ldb, int ldc) {
        libxsmm_gemm_shape brshape = libxsmm_create_gemm_shape(
            BLOCK_M, BLOCK_N, BLOCK_K,
            lda, ldb, ldc,
            /*type A*/LIBXSMM_DATATYPE_F32,
            /*type B*/LIBXSMM_DATATYPE_F32,
            /*type C*/LIBXSMM_DATATYPE_F32,
            /*acctype*/LIBXSMM_DATATYPE_F32
        );
        libxsmm_bitfield brflags = (
            trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE
        ) | (
            trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : LIBXSMM_GEMM_FLAG_NONE
        );
        libxsmm_gemm_batch_reduce_config brconfig;
        memset(&brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config));
        brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

        kernel_func_ = libxsmm_dispatch_brgemm_v2(brshape, brflags, /*prefetch_flags=*/ 0, brconfig);
        memset(&gemm_param_, 0, sizeof(libxsmm_gemm_param));
    }

    void operator()(void* A, void* B, void* C) {
        gemm_param_.a.primary = (void*)A;
        gemm_param_.b.primary = (void*)B;
        gemm_param_.c.primary = (void*)C;
        kernel_func_(&gemm_param_);
    }
private:
    libxsmm_gemmfunction kernel_func_;
    libxsmm_gemm_param gemm_param_;
};

class DotMicroKernel_edge {
public:
    DotMicroKernel_edge(int BLOCK_M, int BLOCK_N, int BLOCK_K, bool trans_a, bool trans_b, int lda, int ldb, int ldc) {
        libxsmm_gemm_shape brshape = libxsmm_create_gemm_shape(
            BLOCK_M, BLOCK_N, BLOCK_K,
            lda, ldb, ldc,
            /*type A*/LIBXSMM_DATATYPE_F32,
            /*type B*/LIBXSMM_DATATYPE_F32,
            /*type C*/LIBXSMM_DATATYPE_F32,
            /*acctype*/LIBXSMM_DATATYPE_F32
        );
        libxsmm_bitfield brflags = (
            trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE
        ) | (
            trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : LIBXSMM_GEMM_FLAG_NONE
        );
        libxsmm_gemm_batch_reduce_config brconfig;
        memset(&brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config));
        brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

        kernel_func_ = libxsmm_dispatch_brgemm_v2(brshape, brflags, /*prefetch_flags=*/ 0, brconfig);
        memset(&gemm_param_, 0, sizeof(libxsmm_gemm_param));
    }

    void operator()(void* A, void* B, void* C) {
        gemm_param_.a.primary = (void*)A;
        gemm_param_.b.primary = (void*)B;
        gemm_param_.c.primary = (void*)C;
        kernel_func_(&gemm_param_);
    }
private:
    libxsmm_gemmfunction kernel_func_;
    libxsmm_gemm_param gemm_param_;
};


template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
using DotMicroKernelRef = std::shared_ptr<DotMicroKernel<BLOCK_M,BLOCK_N,BLOCK_K>>;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
DotMicroKernelRef<BLOCK_M,BLOCK_N,BLOCK_K> create_or_get_dot_microkernel(bool trans_a, bool trans_b, int lda, int ldb, int ldc) {
    thread_local std::unordered_map<DotMicroKernelKey, DotMicroKernelRef<BLOCK_M,BLOCK_N,BLOCK_K>> cache;
    DotMicroKernelKey key(trans_a, trans_b, lda, ldb, ldc);
    auto search = cache.find(key);
    if (search != cache.end()) {
        return search->second;
    } else {
        cache.insert({key, std::make_shared<DotMicroKernel<BLOCK_M,BLOCK_N,BLOCK_K>>(trans_a, trans_b, lda, ldb, ldc)}); 
        return cache[key];
    }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
void dot_tile_update(
    float* A, float* B, float* C,
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc
) {
    auto&& kernel = create_or_get_dot_microkernel<BLOCK_M, BLOCK_N, BLOCK_K>(trans_a, trans_b, lda, ldb, ldc);
    (*kernel)(A, B, C);
}

void dot_edge_update(
    float* A, float* B, float* C,
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc    
){
    auto&& kernel = std::make_shared<DotMicroKernel_edge>(BLOCK_M, BLOCK_N, BLOCK_K, trans_a, trans_b, lda, ldb, ldc);
    (*kernel)(A, B, C);   
}

void zero_fill(float* C, int M, int N, int stride){
    for(int m = 0; m < M; m++){
        memset(C+m*stride, 0, sizeof(float)*N);
    }
}

inline void dequant_(int8_t* B, float* b, __m512 float_zero_point, __m512 float_scale){
    const __m128i b_ = _mm_loadu_si128((const __m128i*)B);
    __m512 vb;
    vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
    vb = _mm512_sub_ps(vb, float_zero_point);
    vb = _mm512_mul_ps(vb, float_scale);
    _mm512_storeu_ps(b, vb); 
}

inline void dequant_(int8_t* B, float* b, __m512 float_zero_point, __m512 float_scale, unsigned short mask){
    const __m128i b_ = _mm_maskz_loadu_epi8(mask, (const __m128i*)B);
    __m512 vb;
    vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
    vb = _mm512_maskz_sub_ps(mask, vb, float_zero_point);
    vb = _mm512_maskz_mul_ps(mask, vb, float_scale);   
    _mm512_mask_storeu_ps(b, mask, vb); 
}



void pack(int8_t *B, int8_t *packedB, int K, int N, int ldb, bool transB) {
    const int blks = (N + BLOCK_N - 1) / BLOCK_N;
    // B is not transposed, shape: K x N
    if (!transB) {
#pragma omp parallel for
        for (int i = 0; i < blks; ++i) {
            int cols = BLOCK_N;        // each time pack BLOCK_N columns
            if (i == blks - 1) {  // last block
                cols = N - i * BLOCK_N;
            }

            const int8_t *psrc = B + i * BLOCK_N;
            int8_t *pdst = packedB + i * K * BLOCK_N;

            for (int r = 0; r < K; ++r) {
                memcpy(pdst, psrc, cols * sizeof(int8_t));
                psrc += ldb;
                pdst += cols;
            }
        }
    }

    //B is transposed, shape: N x K
    else {
#pragma omp parallel for
        for (int i = 0; i < blks; ++i) {
            int rows = BLOCK_N;        // each time pack BLOCK_N elements in N dimension
            if (i == blks - 1) {  // last block
                rows = N - i * BLOCK_N;
            }

            const int8_t *psrc = B + i * BLOCK_N * ldb;
            int8_t *pdst = packedB + i * K * BLOCK_N;

            for (int c = 0; c < K; ++c) {
                for (int r = 0; r < rows; ++r) {
                    pdst[r] = psrc[r * ldb];
                }
                psrc += 1;
                pdst += rows;
            }
        }
    }
}

//per channel
void dequant(int8_t* B, float* b, int K, int N, int32_t* zero_point, float* scale){
    int COLS = N / 16;
    for(int k = 0 ; k < K ; k++){
        int8_t* src = B;
        float* dst = b;  
        int j;
        for(j = 0; j < COLS * 16; j+=16){
            __m512 float_scale = _mm512_loadu_ps(scale+j);
            __m512 float_zero_point = _mm512_castsi512_ps(_mm512_loadu_si512(zero_point+j));       
            dequant_(src, dst, float_zero_point, float_scale);  
            src += 16;
            dst += 16;          
        } 
        if(j < N){
            const int res = N - j;
            unsigned short mask = 0xffff;
            mask = (1 << res) - 1;
            __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale+j);
            __m512 float_zero_point = _mm512_castsi512_ps(_mm512_maskz_loadu_epi32(mask, zero_point+j));  
            dequant_(src, dst, float_zero_point, float_scale, mask);  
        }  
        B += N;
        b += N;
    }
}

//per tensor
void dequant(int8_t* B, float* b, int K, int N, int32_t zero_point, float scale){
    __m512 float_scale = _mm512_set1_ps(scale);
    __m512 float_zero_point = _mm512_set1_ps((const float)zero_point);    
    int COLS = N / 16;
    for(int k = 0 ; k < K ; k++){
        int8_t* src = B;
        float* dst = b;
        int j;  
        for(j = 0; j < COLS * 16; j+=16){        
            dequant_(src, dst, float_zero_point, float_scale);  
            src += 16;
            dst += 16;          
        }
        if(j < N){ //elements < 16
            const int res = N - j;
            unsigned short mask = 0xffff;
            mask = (1 << res) - 1;
            dequant_(src, dst, float_zero_point, float_scale, mask);  
        }    
        B += N;
        b += N;
    }
}

void add_bias(float* C, float* bias, int M, int N, int ldc){
    int COLS = N / 16;
    int j;
    for(j = 0; j < COLS*16; j+=16){
        __m512 float_bias = _mm512_loadu_ps(bias+j);
        float* c = C + j; 
        for(int m = 0; m < M; m++){
            __m512 vc = _mm512_loadu_ps(c);
            vc= _mm512_add_ps(vc, float_bias);
            _mm512_storeu_ps(c, vc); 
            c += ldc;
        }
    }
    if(j < N){
        const int res = N - j;
        unsigned short mask = 0xffff;
        mask = (1 << res) - 1;
        __m512 float_bias = _mm512_maskz_loadu_ps(mask, bias+j);
        float* c = C + j;
        for(int m = 0; m < M; m++){
            __m512 vc = _mm512_maskz_loadu_ps(mask, c);   
            vc = _mm512_mask_add_ps(vc, mask, vc, float_bias);
            _mm512_mask_storeu_ps(c, mask, vc); 
            c += ldc;
        }
    }
}

//per channel
void woq_gemm(float* A, int8_t* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, int32_t* zero_point, float* scale, float* bias){
#define PTR_OFFSET(base, offset0, offset1, stride0)\
    (base) + (offset0)*(stride0) + (offset1)

    //const int BLOCK_M = 4, BLOCK_N = 64, BLOCK_K = 1024; //BLOCK_N must a multiple of 16
    const int MB = (M + BLOCK_M -1)/BLOCK_M, NB = (N + BLOCK_N - 1)/BLOCK_N, KB = (K + BLOCK_K -1)/BLOCK_K;

    #pragma omp parallel for collapse(2)
    for(int mb = 0 ; mb < MB; mb++){
        for(int nb = 0 ; nb < NB; nb++){
            int mb_start = mb * BLOCK_M;
            int m_bs = std::min(BLOCK_M, M-mb_start);
            int nb_start = nb * BLOCK_N;
            int n_bs = std::min(BLOCK_N, N-nb_start);
            float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
            zero_fill(C_offset, m_bs, n_bs, ldc);
            float* bi_offset = (float *)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));             
            for(int kb = 0; kb < KB; kb++){
                int kb_start = kb * BLOCK_K;
                int k_bs = std::min(BLOCK_K, K-kb_start);
                float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
                int8_t* B_offset = B + nb_start*K + kb_start*n_bs;
                dequant(B_offset, bi_offset, k_bs, n_bs, zero_point+nb_start, scale+nb_start);
                if(m_bs == BLOCK_M && n_bs == BLOCK_N && k_bs == BLOCK_K){
                    dot_tile_update<BLOCK_N,BLOCK_M,BLOCK_K>(
                        bi_offset,
                        A_offset,
                        C_offset,
                        false, false,
                        BLOCK_N, lda, ldc
                    );                     
                }
                else{
                    dot_edge_update(
                        bi_offset, 
                        A_offset,
                        C_offset,
                        n_bs,
                        m_bs,
                        k_bs,
                        false, false,
                        n_bs, lda, ldc
                    );
                }
            }
            add_bias(C_offset, bias+nb_start, m_bs, n_bs, ldc);
            free(bi_offset);
        }
    }
}


//per tensor
void woq_gemm(float* A, int8_t* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, int32_t zero_point, float scale, float* bias){
#define PTR_OFFSET(base, offset0, offset1, stride0)\
    (base) + (offset0)*(stride0) + (offset1)

    //const int BLOCK_M = 4, BLOCK_N = 64, BLOCK_K = 512; //BLOCK_N must a multiple of 64
    const int MB = (M + BLOCK_M -1)/BLOCK_M, NB = (N + BLOCK_N - 1)/BLOCK_N, KB = (K + BLOCK_K -1)/BLOCK_K;  //num of blks


    #pragma omp parallel for collapse(2)
    for(int mb = 0 ; mb < MB; mb++){
        for(int nb = 0 ; nb < NB; nb++){
            int mb_start = mb * BLOCK_M;
            int m_bs = std::min(BLOCK_M, M-mb_start);
            int nb_start = nb * BLOCK_N;
            int n_bs = std::min(BLOCK_N, N-nb_start);
            float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
            zero_fill(C_offset, m_bs, n_bs, ldc);
            float* bi_offset = (float *)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));             
            for(int kb = 0; kb < KB; kb++){
                int kb_start = kb * BLOCK_K;
                int k_bs = std::min(BLOCK_K, K-kb_start);
                float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
                int8_t* B_offset = B + nb_start*K + kb_start*n_bs;
                dequant(B_offset, bi_offset, k_bs, n_bs, zero_point, scale);
                if(m_bs == BLOCK_M && n_bs == BLOCK_N && k_bs == BLOCK_K){
                    dot_tile_update<BLOCK_N,BLOCK_M,BLOCK_K>(
                        bi_offset,
                        A_offset,
                        C_offset,
                        false, false,
                        BLOCK_N, lda, ldc
                    );                     
                }
                else{
                    dot_edge_update(
                        bi_offset, 
                        A_offset,
                        C_offset,
                        n_bs,
                        m_bs,
                        k_bs,
                        false, false,
                        n_bs, lda, ldc
                    );
                }                  
            }
            add_bias(C_offset, bias+nb_start, m_bs, n_bs, ldc);
            free(bi_offset);              
        }
    }   
}


#endif

