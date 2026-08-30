#pragma once

#include <string>

#include "blackforge/blackbit/checkpoint.hpp"
#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"
#include "blackforge/blackbit/cuda_model.hpp"

namespace blackforge::blackbit::cuda {

// Uses the same BFBIT v1 on-disk representation as the CPU path: ternary
// parameters remain packed, dense parameters remain FP32, and optimizer
// moments are named low-rank buffers. CPU and CUDA checkpoints interoperate.
void saveCheckpoint(const std::string& path, BlackBitModel& model,
                    const BlackBitTrainingState& state,
                    LowRankProjectedOptimizer* optimizer = nullptr);

BlackBitTrainingState loadCheckpoint(const std::string& path, BlackBitModel& model,
                                     LowRankProjectedOptimizer* optimizer = nullptr);

}  // namespace blackforge::blackbit::cuda
