#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

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

/// Wrapper seguro para hipMemcpy (síncrono) con validación y retry
/// B3.63 FIX: Añade manejo robusto de illegal memory access
inline bool safe_hipMemcpy(
    void* dst,
    const void* src,
    size_t bytes,
    hipMemcpyKind kind,
    const char* debug_name = "unknown") {
  
  if (!dst || !src || bytes == 0) {
    std::cerr << "[D2H SAFE] Skip sync copy " << debug_name
              << " - null ptr or zero bytes\n";
    return false;
  }
  
  // B3.63 FIX: Retry logic para handle illegal memory access races
  constexpr int MAX_RETRIES = 3;
  for (int retry = 0; retry < MAX_RETRIES; ++retry) {
    hipError_t err = hipMemcpy(dst, src, bytes, kind);
    if (err == hipSuccess) {
      return true;
    }
    
    if (err == hipErrorIllegalAddress) {
      std::cerr << "[D2H SAFE] Illegal memory access on " << debug_name
                << " (attempt " << (retry + 1) << "/" << MAX_RETRIES << ")\n";
      // Small delay to allow GPU state to stabilize
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    
    std::cerr << "[D2H SAFE] Sync copy failed " << debug_name << ": "
              << hipGetErrorString(err) << "\n";
    return false;
  }
  
  std::cerr << "[D2H SAFE] Sync copy failed after retries " << debug_name << "\n";
  return false;
}

} // namespace greta_d2h_safe
