#ifndef _GGML_H
#define _GGML_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_FILE_VERSION 2

#define GGML_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#define GGML_MAX_CONTEXTS       64
#define GGML_MAX_SRC            10
#define GGML_MAX_N_THREADS      512
#define GGML_MAX_OP_PARAMS      64

#ifndef GGML_MAX_NAME
#   define GGML_MAX_NAME        64
#endif

#define GGML_DEFAULT_N_THREADS  4
#define GGML_DEFAULT_GRAPH_SIZE 2048

#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

#define GGML_EXIT_SUCCESS 0
#define GGML_EXIT_ABORTED 1

#define GGML_ROPE_TYPE_NEOX 2

#define GGUF_MAGIC "GGUF"

#define GGUF_VERSION 3

#define GGUF_DEFAULT_ALIGNMENT 32

#define GGML_UNUSED(x) (void)(x)

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
#   define GGML_UNREACHABLE() do { fprintf(stderr, "statement should be unreachable\n"); abort(); } while(0)
#elif defined(__GNUC__)
#   define GGML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#   define GGML_UNREACHABLE() __assume(0)
#else
#   define GGML_UNREACHABLE() ((void) 0)
#endif

#ifdef __cplusplus
#   define GGML_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#   define GGML_NORETURN __declspec(noreturn)
#else
#   define GGML_NORETURN _Noreturn
#endif

#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

#ifdef  __cplusplus
extern "C" {
#endif

    GGML_NORETURN GGML_ATTRIBUTE_FORMAT(3, 4)
    GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);

    enum ggml_status {
        GGML_STATUS_ALLOC_FAILED = -2,
        GGML_STATUS_FAILED = -1,
        GGML_STATUS_SUCCESS = 0,
        GGML_STATUS_ABORTED = 1,
    };

    // get ggml_status name string
    GGML_API const char * ggml_status_to_string(enum ggml_status status);

    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        GGML_TYPE_Q4_0_4_4 = 31,
        GGML_TYPE_Q4_0_4_8 = 32,
        GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        GGML_TYPE_COUNT,
    };

    enum ggml_backend_type {
        GGML_BACKEND_TYPE_CPU = 0,
        GGML_BACKEND_TYPE_GPU = 10,
        GGML_BACKEND_TYPE_GPU_SPLIT = 20,
    };

    // model file types
    enum ggml_ftype {
        GGML_FTYPE_UNKNOWN        = -1,
        GGML_FTYPE_ALL_F32        = 0,
        GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
        GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0_4_4 = 25, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0_4_8 = 26, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0_8_8 = 27, // except 1d tensors
    };

    // available tensor operations:
    enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_COUNT_EQUAL,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_IM2COL_BACK,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_POOL_2D_BACK,
        GGML_OP_UPSCALE, // nearest interpolate
        GGML_OP_PAD,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_LEAKY_RELU,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_RWKV_WKV,

        GGML_OP_UNARY,

        GGML_OP_MAP_UNARY,
        GGML_OP_MAP_BINARY,

        GGML_OP_MAP_CUSTOM1_F32,
        GGML_OP_MAP_CUSTOM2_F32,
        GGML_OP_MAP_CUSTOM3_F32,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        GGML_OP_OPT_STEP_ADAMW,

        GGML_OP_COUNT,
    };

    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,

        GGML_UNARY_OP_COUNT,
    };

    enum ggml_object_type {
        GGML_OBJECT_TYPE_TENSOR,
        GGML_OBJECT_TYPE_GRAPH,
        GGML_OBJECT_TYPE_WORK_BUFFER
    };

    enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_INFO  = 1,
        GGML_LOG_LEVEL_WARN  = 2,
        GGML_LOG_LEVEL_ERROR = 3,
        GGML_LOG_LEVEL_DEBUG = 4,
        GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };

    // this tensor...
    enum ggml_tensor_flag {
        GGML_TENSOR_FLAG_INPUT  =  1, // ...is an input for the GGML compute graph
        GGML_TENSOR_FLAG_OUTPUT =  2, // ...is an output for the GGML compute graph
        GGML_TENSOR_FLAG_PARAM  =  4, // ...contains trainable parameters
        GGML_TENSOR_FLAG_LOSS   =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
    };

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type; 

        GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor");

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * grad;
        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        // char padding[4];
    };

    struct ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);

#ifdef  __cplusplus
}
#endif

#endif
