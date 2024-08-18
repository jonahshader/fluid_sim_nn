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
  Soil(const glm::vec2 &region, float kernel_size, float wall_size);

  void add(const SoilParticle &p);

  const std::vector<SoilCell> &get_soil_grid() const { return soil_grid; }
  const std::vector<bool> &get_wall_grid() const { return wall_grid; }
  const std::vector<SoilCell> get_neighbors(const glm::vec2 &pos) const;
  float get_soil_cell_size() const { return soil_cell_size; }
  float get_wall_cell_size() const { return wall_cell_size; }
  int get_soil_grid_width() const { return soil_grid_width; }
  int get_soil_grid_height() const { return soil_grid_height; }
  int get_wall_grid_width() const { return wall_grid_width; }
  int get_wall_grid_height() const { return wall_grid_height; }
  bool get_wall(const glm::vec2 &pos) const;
  void set_wall(const glm::vec2 &pos, bool wall);
  void populate_walls(float p, std::mt19937 &gen);

private:
  std::vector<SoilCell> soil_grid{};
  std::vector<bool> wall_grid{};
  float soil_cell_size;
  float wall_cell_size;
  int soil_grid_width{};
  int soil_grid_height{};
  int wall_grid_width{};
  int wall_grid_height{};

  SoilCell *get_cell(const glm::vec2 &pos);

  std::optional<int> get_soil_cell_index(const glm::vec2 &pos) const;
  std::optional<int> get_wall_cell_index(const glm::vec2 &pos) const;
};
