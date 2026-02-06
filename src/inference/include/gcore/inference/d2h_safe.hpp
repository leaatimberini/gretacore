#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>

/// Namespace para wrappers seguros de operaciones D2H (Device-to-Host)
namespace greta_d2h_safe {

/// Wrapper seguro para hipMemcpyAsync D2H con sync guarantee
/// B3.63: Añade sincronización antes y después de la copia para evitar races
inline bool safe_hipMemcpyAsync(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    hipStream_t stream,
    const char* debug_name = "unknown") {
  
  if (!dst || !src || bytes == 0) {
    std::cerr << "[D2H SAFE] Skip async copy " << debug_name
              << " - null ptr or zero bytes\n";
    return false;
  }
  
  // B3.63 FIX: Sincronizar stream antes de copiar
  hipError_t sync_err = hipStreamSynchronize(stream);
  if (sync_err != hipSuccess) {
    std::cerr << "[D2H SAFE] Stream sync failed before " << debug_name << ": "
              << hipGetErrorString(sync_err) << "\n";
    return false;
  }
  
  hipError_t err = hipMemcpyAsync(dst, src, bytes, kind, stream);
  if (err != hipSuccess) {
    std::cerr << "[D2H SAFE] Async copy failed " << debug_name << ": "
              << hipGetErrorString(err) << "\n";
    return false;
  }
  
  // Sincronizar después para garantizar que la copia completó
  sync_err = hipStreamSynchronize(stream);
  if (sync_err != hipSuccess) {
    std::cerr << "[D2H SAFE] Stream sync failed after " << debug_name << ": "
              << hipGetErrorString(sync_err) << "\n";
    return false;
  }
  
  return true;
}

} // namespace greta_d2h_safe
