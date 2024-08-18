#include "soil.h"

#include <cmath>

constexpr float MAX_SOIL_PARTICLE_RADIUS = 2.0f;

Soil::Soil(const glm::vec2 &region, float kernel_size) : cell_size(kernel_size + MAX_SOIL_PARTICLE_RADIUS)
{
  grid_width = static_cast<int>(std::ceil(region.x / cell_size));
  grid_height = static_cast<int>(std::ceil(region.y / cell_size));
  auto grid_size = grid_width * grid_height;
  // populate empty grid
  for (auto i = 0; i < grid_size; ++i)
  {
    soil_grid.emplace_back();
    wall_grid.push_back(false);
  }
}

void Soil::add(const SoilParticle &p)
{
  auto cell = get_cell(p.pos);
  if (cell)
  {
    cell->push_back(p);
  }
}

SoilCell *Soil::get_cell(const glm::vec2 &pos)
{
  auto index = get_cell_index(pos);
  return index ? &soil_grid[*index] : nullptr;
}

std::optional<int> Soil::get_cell_index(const glm::vec2 &pos) const
{
  int cell_x = static_cast<int>(std::floor(pos.x / cell_size));
  int cell_y = static_cast<int>(std::floor(pos.y / cell_size));
  if (cell_x < 0 || cell_x >= grid_width || cell_y < 0 || cell_y >= grid_height)
  {
    return std::nullopt;
  }
  else
  {
    return cell_y * grid_width + cell_x;
  }
}

bool Soil::get_wall(const glm::vec2 &pos) const
{
  auto index = get_cell_index(pos);
  return index ? wall_grid[*index] : false;
}

void Soil::set_wall(const glm::vec2 &pos, bool wall)
{
  auto index = get_cell_index(pos);
  if (index)
  {
    wall_grid[*index] = wall;
  }
}

void Soil::populate_walls(float p, std::mt19937 &gen)
{
  std::bernoulli_distribution dist(p);
  for (auto i = 0; i < wall_grid.size(); ++i)
  {
    wall_grid[i] = dist(gen);
  }
}

const std::vector<SoilCell> Soil::get_neighbors(const glm::vec2 &pos) const
{
  std::vector<SoilCell> neighbors;
  int cell_x = static_cast<int>(std::floor(pos.x / cell_size));
  int cell_y = static_cast<int>(std::floor(pos.y / cell_size));
  for (int y = -1; y <= 1; ++y)
  {
    for (int x = -1; x <= 1; ++x)
    {
      int nx = cell_x + x;
      int ny = cell_y + y;
      if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height)
      {
        neighbors.push_back(soil_grid[ny * grid_width + nx]);
      }
    }
  }
  return neighbors;
}
