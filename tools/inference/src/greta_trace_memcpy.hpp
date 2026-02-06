/**
 * @file greta_trace_memcpy.hpp
 * @brief Instrumentación para auditoría de transferencias HIP D2H
 * 
 * B3.62: HIP D2H Transfer Audit
 * Header para tracing de hipMemcpy/hipMemcpyAsync Device→Host
 * 
 * NOTE: Designed to be compiled with HIP/ROCm toolchain (MI300X)
 */

#ifndef GRETA_TRACE_MEMCPY_HPP
#define GRETA_TRACE_MEMCPY_HPP

// Forward declare HIP types when not available
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
typedef enum hipMemcpyKind { hipMemcpyDeviceToHost } hipMemcpyKind;
typedef struct hipStream_st* hipStream_t;
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef GRETA_TRACE_MEMCPY

#define GRETA_TRACE_MEMCPY_ENABLED 1

/**
 * @brief Macro para trazar llamadas D2H - Versión simplified
 */
#define GRETA_TRACE_D2H_BEFORE(tensor_name, src, bytes, stream) do { \
    fprintf(stderr, "[D2H TRACE] %s:before\n", tensor_name); \
    fprintf(stderr, "  src_ptr=%p\n", (void*)(src)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
    fprintf(stderr, "  stream=%p\n", (void*)(stream)); \
    /*hipError_t _sync_err = hipStreamSynchronize(stream);*/ \
    /*if (_sync_err != hipSuccess) { */\
    /*    fprintf(stderr, "[D2H ERROR] pre-sync failed: %s\n", hipGetErrorString(_sync_err)); */\
    /*} */\
    /*hipError_t _pre_err = hipGetLastError(); */\
    /*fprintf(stderr, "  pre-copy hipGetLastError: %s\n", hipGetErrorString(_pre_err)); */\
} while(0)

#define GRETA_TRACE_D2H_AFTER(tensor_name, dst, bytes, stream) do { \
    fprintf(stderr, "[D2H TRACE] %s:after\n", tensor_name); \
    fprintf(stderr, "  dst_ptr=%p\n", (void*)(dst)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
} while(0)

#define GRETA_TRACE_D2H_ERROR(tensor_name, src, dst, bytes, err) do { \
    fprintf(stderr, "[D2H FATAL] %s:FAILED\n", tensor_name); \
    fprintf(stderr, "  src_ptr=%p\n", (void*)(src)); \
    fprintf(stderr, "  dst_ptr=%p\n", (void*)(dst)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
    fprintf(stderr, "  hipError=%s\n", hipGetErrorString(err)); \
    fprintf(stderr, "[D2H FATAL] Aborting...\n"); \
    abort(); \
} while(0)

/**
 * @brief Wrapper seguro para hipMemcpy D2H con tracing
 */
inline bool greta_trace_hipMemcpy(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    const char* tensor_name
) {
    if (kind != hipMemcpyDeviceToHost) {
        return hipMemcpy(dst, src, bytes, kind) == hipSuccess;
    }
    
    GRETA_TRACE_D2H_BEFORE(tensor_name, src, bytes, nullptr);
    
    hipError_t err = hipMemcpy(dst, src, bytes, kind);
    
    if (err != hipSuccess) {
        GRETA_TRACE_D2H_ERROR(tensor_name, src, dst, bytes, err);
        return false;
    }
    
    GRETA_TRACE_D2H_AFTER(tensor_name, dst, bytes, nullptr);
    return true;
}

/**
 * @brief Wrapper seguro para hipMemcpyAsync D2H con tracing
 */
inline bool greta_trace_hipMemcpyAsync(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    hipStream_t stream,
    const char* tensor_name
) {
    if (kind != hipMemcpyDeviceToHost) {
        return hipMemcpyAsync(dst, src, bytes, kind, stream) == hipSuccess;
    }
    
    GRETA_TRACE_D2H_BEFORE(tensor_name, src, bytes, stream);
    
    hipError_t err = hipMemcpyAsync(dst, src, bytes, kind, stream);
    
    if (err != hipSuccess) {
        GRETA_TRACE_D2H_ERROR(tensor_name, src, dst, bytes, err);
        return false;
    }
    
    GRETA_TRACE_D2H_AFTER(tensor_name, dst, bytes, stream);
    return true;
}

#else

#define GRETA_TRACE_MEMCPY_ENABLED 0

inline bool greta_trace_hipMemcpy(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    const char* tensor_name
) {
    (void)tensor_name;
    return hipMemcpy(dst, src, bytes, kind) == hipSuccess;
}

inline bool greta_trace_hipMemcpyAsync(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    hipStream_t stream,
    const char* tensor_name
) {
    (void)tensor_name;
    return hipMemcpyAsync(dst, src, bytes, kind, stream) == hipSuccess;
}

#endif // GRETA_TRACE_MEMCPY

// Macros de conveniencia
#ifdef GRETA_TRACE_MEMCPY
#define GRETA_MEMCPY_D2H(dst, src, bytes, name) \
    greta_trace_hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost, name)
#define GRETA_MEMCPY_D2H_ASYNC(dst, src, bytes, stream, name) \
    greta_trace_hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream, name)
#else
#define GRETA_MEMCPY_D2H(dst, src, bytes, name) \
    (hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost) == hipSuccess)
#define GRETA_MEMCPY_D2H_ASYNC(dst, src, bytes, stream, name) \
    (hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream) == hipSuccess)
#endif

#endif // GRETA_TRACE_MEMCPY_HPP
