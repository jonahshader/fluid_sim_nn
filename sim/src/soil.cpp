#include "soil.h"

#include <cmath>

constexpr float MAX_SOIL_PARTICLE_RADIUS = 32.0f;

Soil::Soil(const glm::vec2 &region, float kernel_size) : cell_size(kernel_size + MAX_SOIL_PARTICLE_RADIUS)
{
  grid_width = static_cast<int>(std::ceil(region.x / cell_size));
  grid_height = static_cast<int>(std::ceil(region.y / cell_size));
  auto grid_size = grid_width * grid_height;
  // populate empty grid
  for (auto i = 0; i < grid_size; ++i)
  {
    grid.emplace_back();
  }
}

void Soil::add(const SoilParticle &p)
{
  // TODO
}

std::optional<std::vector<SoilParticle> &> Soil::get_cell(const glm::vec2 &pos)
{
  int cell_x = static_cast<int>(std::floor(pos.x / cell_size));
  int cell_y = static_cast<int>(std::floor(pos.y / cell_size));
  if (cell_x < 0 || cell_x >= grid_width || cell_y < 0 || cell_y >= grid_height)
  {
    return std::nullopt;
  }
}