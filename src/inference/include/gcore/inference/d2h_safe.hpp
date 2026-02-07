#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>
#include <cstdlib>

/// Namespace para wrappers seguros de operaciones D2H (Device-to-Host)
namespace greta_d2h_safe {

/// Debug mode flag - sincronizaciones agresivas cuando GRETA_D2H_DEBUG=1
inline bool is_debug_mode() {
    const char* v = std::getenv("GRETA_D2H_DEBUG");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

/// Metadata para trazas de D2H
struct D2HMetadata {
    const char* tensor_name = "unknown";
    int step = -1;
    int layer = -1;
    size_t offset_bytes = 0;
    size_t size_bytes = 0;
    size_t alloc_bytes = 0;
};

/// Wrapper seguro para hipMemcpyAsync D2H con validación completa
/// B3.64: Añade instrumentación de debug para diagnosis de illegal memory access
inline bool greta_hip_memcpy_d2h_safe(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    hipStream_t stream,
    const D2HMetadata& meta = D2HMetadata()) {

    // Requisito 1: Validación de punteros
    if (!dst || !src || bytes == 0) {
        std::cerr << "[D2H_ERROR] null pointer detected: "
                  << "dst=" << dst << " src=" << src << " bytes=" << bytes
                  << " tensor=" << meta.tensor_name << "\n";
        return false;
    }

    // Requisito 2: Validación de rangos
    if (meta.alloc_bytes > 0) {
        if (meta.offset_bytes + meta.size_bytes > meta.alloc_bytes) {
            std::cerr << "[D2H_BOUNDS] tensor=" << meta.tensor_name
                      << " offset=" << meta.offset_bytes
                      << " size=" << meta.size_bytes
                      << " alloc=" << meta.alloc_bytes << "\n";
            return false;
        }
    }

    // Requisito 3: Log estructurado antes del memcpy
    std::cerr << "[D2H_CHECK] tensor=" << meta.tensor_name
              << " step=" << meta.step
              << " layer=" << meta.layer
              << " src_ptr=" << reinterpret_cast<uintptr_t>(src)
              << " dst_ptr=" << reinterpret_cast<uintptr_t>(dst)
              << " offset=" << meta.offset_bytes
              << " size=" << meta.size_bytes
              << " alloc=" << meta.alloc_bytes << "\n";

    // Requisito 4: Debug mode con sincronizaciones agresivas
    if (is_debug_mode()) {
        // Sincronizar stream antes
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            std::cerr << "[D2H_DEBUG] stream sync failed before: "
                      << hipGetErrorString(sync_err) << "\n";
        }
    }

    // Ejecutar el memcpy
    hipError_t err = hipMemcpyAsync(dst, src, bytes, kind, stream);
    
    if (err != hipSuccess) {
        std::cerr << "[D2H_ERROR] hipMemcpyAsync failed: "
                  << hipGetErrorString(err)
                  << " tensor=" << meta.tensor_name << "\n";
        
        if (is_debug_mode()) {
            // HipDeviceSynchronize al detectar error
            hipError_t dev_sync = hipDeviceSynchronize();
            if (dev_sync != hipSuccess) {
                std::cerr << "[D2H_DEBUG] hipDeviceSynchronize error: "
                          << hipGetErrorString(dev_sync) << "\n";
            }
        }
        return false;
    }

    if (is_debug_mode()) {
        // Sincronizar después del memcpy
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            std::cerr << "[D2H_DEBUG] stream sync failed after: "
                      << hipGetErrorString(sync_err) << "\n";
        }
    }

    return true;
}

/// Wrapper seguro para hipMemcpy (síncrono) con validación completa
/// B3.64: Añade instrumentación de debug para diagnosis de illegal memory access
inline bool greta_hip_memcpy_d2h_safe_sync(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    const D2HMetadata& meta = D2HMetadata()) {

    // Requisito 1: Validación de punteros
    if (!dst || !src || bytes == 0) {
        std::cerr << "[D2H_ERROR] null pointer detected: "
                  << "dst=" << dst << " src=" << src << " bytes=" << bytes
                  << " tensor=" << meta.tensor_name << "\n";
        return false;
    }

    // Requisito 2: Validación de rangos
    if (meta.alloc_bytes > 0) {
        if (meta.offset_bytes + meta.size_bytes > meta.alloc_bytes) {
            std::cerr << "[D2H_BOUNDS] tensor=" << meta.tensor_name
                      << " offset=" << meta.offset_bytes
                      << " size=" << meta.size_bytes
                      << " alloc=" << meta.alloc_bytes << "\n";
            return false;
        }
    }

    // Requisito 3: Log estructurado antes del memcpy
    std::cerr << "[D2H_SAFE_WRAPPER] engaged for tensor=" << meta.tensor_name << "\n";
    std::cerr << "[D2H_CHECK] tensor=" << meta.tensor_name
              << " step=" << meta.step
              << " layer=" << meta.layer
              << " src_ptr=" << reinterpret_cast<uintptr_t>(src)
              << " dst_ptr=" << reinterpret_cast<uintptr_t>(dst)
              << " offset=" << meta.offset_bytes
              << " size=" << meta.size_bytes
              << " alloc=" << meta.alloc_bytes << "\n";

    // Requisito 4: Debug mode con sincronizaciones agresivas
    if (is_debug_mode()) {
        hipError_t dev_sync = hipDeviceSynchronize();
        if (dev_sync != hipSuccess) {
            std::cerr << "[D2H_DEBUG] pre-sync error: "
                      << hipGetErrorString(dev_sync) << "\n";
        }
    }

    // Retry logic para handle illegal memory access races
    constexpr int MAX_RETRIES = 3;
    for (int retry = 0; retry < MAX_RETRIES; ++retry) {
        hipError_t err = hipMemcpy(dst, src, bytes, kind);
        if (err == hipSuccess) {
            if (is_debug_mode()) {
                hipError_t dev_sync = hipDeviceSynchronize();
                if (dev_sync != hipSuccess) {
                    std::cerr << "[D2H_DEBUG] post-sync error: "
                              << hipGetErrorString(dev_sync) << "\n";
                }
            }
            return true;
        }
        
        if (err == hipErrorIllegalAddress) {
            std::cerr << "[D2H_ERROR] Illegal memory access on "
                      << meta.tensor_name
                      << " (attempt " << (retry + 1) << "/" << MAX_RETRIES << ")\n";
            if (is_debug_mode()) {
                hipError_t dev_sync = hipDeviceSynchronize();
                if (dev_sync != hipSuccess) {
                    std::cerr << "[D2H_DEBUG] recovery sync error: "
                              << hipGetErrorString(dev_sync) << "\n";
                }
            }
            // Small delay to allow GPU state to stabilize
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        std::cerr << "[D2H_ERROR] Sync copy failed "
                  << meta.tensor_name << ": "
                  << hipGetErrorString(err) << "\n";
        return false;
    }

    std::cerr << "[D2H_ERROR] Sync copy failed after retries "
              << meta.tensor_name << "\n";
    return false;
}

} // namespace greta_d2h_safe

// Aliases para compatibilidad
namespace greta_d2h_safe {
    inline bool safe_hipMemcpy(
        void* dst,
        const void* src,
        size_t bytes,
        hipMemcpyKind kind,
        const char* debug_name = "unknown") {
        D2HMetadata meta;
        meta.tensor_name = debug_name;
        return greta_hip_memcpy_d2h_safe_sync(dst, src, bytes, kind, meta);
    }

    inline bool safe_hipMemcpyAsync(
        void* dst,
        const void* src,
        size_t bytes,
        hipMemcpyKind kind,
        hipStream_t stream,
        const char* debug_name = "unknown") {
        D2HMetadata meta;
        meta.tensor_name = debug_name;
        return greta_hip_memcpy_d2h_safe(dst, src, bytes, kind, stream, meta);
    }
} // namespace greta_d2h_safe
