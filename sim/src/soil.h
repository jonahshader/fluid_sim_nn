#pragma once

#include <vector>

#include "soil_particle.h"

using SoilCell = std::vector<SoilParticle>;

class Soil
{
  // soil is essentially a bunch of slow-moving or fixed circles which the water adheres to,
  // producing the capillary effect.
public:
  Soil(const glm::vec2 &region, float kernel_size);

  void add(const SoilParticle &p);

  const std::vector<SoilCell> &get_grid() const { return grid; }
  const std::vector<SoilCell> get_neighbors(const glm::vec2 &pos) const;
  float get_cell_size() const { return cell_size; }
  int get_grid_width() const { return grid_width; }
  int get_grid_height() const { return grid_height; }

private:
  std::vector<SoilCell> grid{};
  float cell_size;
  int grid_width{};
  int grid_height{};

  SoilCell *get_cell(const glm::vec2 &pos);
};