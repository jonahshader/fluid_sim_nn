#pragma once

#include <filesystem>
#include <vector>
#include <optional>
#include <random>

#include "soil_particle.h"

using SoilCell = std::vector<SoilParticle>;

class Soil
{
  // soil is essentially a bunch of slow-moving or fixed circles which the water adheres to,
  // producing the capillary effect.
  // also contains walls, which are solid squares.
public:
  Soil(const glm::vec2 &region, float kernel_size);

  void add(const SoilParticle &p);

  const std::vector<SoilCell> &get_soil_grid() const { return soil_grid; }
  const std::vector<bool> &get_wall_grid() const { return wall_grid; }
  const std::vector<SoilCell> get_neighbors(const glm::vec2 &pos) const;
  float get_cell_size() const { return cell_size; }
  int get_grid_width() const { return grid_width; }
  int get_grid_height() const { return grid_height; }
  bool get_wall(const glm::vec2 &pos) const;
  void set_wall(const glm::vec2 &pos, bool wall);
  void populate_walls(float p, std::mt19937 &gen);

private:
  std::vector<SoilCell> soil_grid{};
  std::vector<bool> wall_grid{};
  float cell_size;
  int grid_width{};
  int grid_height{};

  SoilCell *get_cell(const glm::vec2 &pos);

  std::optional<int> get_cell_index(const glm::vec2 &pos) const;
};
