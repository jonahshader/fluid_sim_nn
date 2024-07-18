#pragma once

#include "glm/glm.hpp"

struct SoilParticle
{
  glm::vec2 pos{};
  float radius{16.0f};
  float adhesion{16000.0f};    // max acceleration at adhesion_radius limits
  float adhesion_radius{3.0f}; // radius +- this value is affected by adhesion
};