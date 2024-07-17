#pragma once

#include <vector>
#include <optional>

#include "soil_particle.h"

class Soil
{
  // soil is essentially a bunch of slow-moving or fixed circles which the water adheres to,
  // producing the capillary effect.
public:
  Soil(const glm::vec2 &region, float kernel_size);

  void add(const SoilParticle &p);

private:
  std::vector<std::vector<SoilParticle>> grid{};
  float cell_size;
  int grid_width{};
  int grid_height{};

  std::optional<std::vector<SoilParticle> &> get_cell(const glm::vec2 &pos);
};