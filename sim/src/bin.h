#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "glm/glm.hpp"

struct Bin
{
  glm::vec2 vel{0.0f, 0.0f};
  int particles{0};
  float avg_vel{0.0f};
  float kinetic_energy{0.0f};
  float density{0.0f};
  float soil_sand_density{0.0f};
  float soil_silt_density{0.0f};
  float soil_clay_density{0.0f};

  void compute_average(int samples)
  {
    if (particles > 0)
    {
      vel /= particles;
      avg_vel /= particles;
      kinetic_energy /= particles;
    }
    if (samples > 0)
    {
      density /= samples;
    }
  }
};

struct Bins
{
  float bin_size;

  glm::vec2 start;
  int width;
  int height;
  int samples{0};
  std::vector<Bin> bins{};

  void compute_averages()
  {
    for (auto &bin : bins)
    {
      bin.compute_average(samples);
    }
  }

  void reset()
  {
    for (auto &bin : bins)
    {
      bin = Bin{};
    }
    samples = 0;
  }

  void print_stats()
  {
    // find max density, max velocity, max kinetic energy
    float max_density = 0.0f;
    float max_avg_vel = 0.0f;
    float max_ke = 0.0f;
    glm::vec2 max_vel{0.0f, 0.0f};
    for (const auto &bin : bins)
    {
      max_density = std::max(max_density, bin.density);
      max_avg_vel = std::max(max_avg_vel, glm::length(bin.vel));
      max_ke = std::max(max_ke, bin.kinetic_energy);
      if (glm::length(bin.vel) > glm::length(max_vel))
      {
        max_vel = bin.vel;
      }
    }

    std::cout << "Max density: " << max_density << std::endl;
    std::cout << "Max velocity: " << max_avg_vel << std::endl;
    std::cout << "Max kinetic energy: " << max_ke << std::endl;
    std::cout << "Max velocity: " << max_vel.x << ", " << max_vel.y << std::endl;
  }

  void write_metadata_json(const std::filesystem::path &path)
  {
    std::ofstream file(path / "metadata.json");
    file << "{\n";
    file << "  \"bin_size\": " << bin_size << ",\n";
    file << "  \"start\": [" << start.x << ", " << start.y << "],\n";
    file << "  \"width\": " << width << ",\n";
    file << "  \"height\": " << height << ",\n";
    // file << "  \"attributes\": [\"vel_x\", \"vel_y\", \"avg_vel\", \"kinetic_energy\", \"density\", \"soil_sand_density\", \"soil_silt_density\", \"soil_clay_density\"]\n";
    // skip soil data for now
    file << "  \"attributes\": [\"vel_x\", \"vel_y\", \"avg_vel\", \"kinetic_energy\", \"density\"]\n";
    file << "}\n";
  }

  void write_to_file(const std::filesystem::path &path, int frame)
  {
    // the channels we want to record are vel, avg_vel, kinetic_energy, density

    // write the data matching pytorch conv2d input
    // (batch, channels, height, width)
    // batch = 1, channels = 6, height = bins.height, width = bins.width

    std::ofstream file(path / ("frame_" + std::to_string(frame) + ".bin"), std::ios::binary);
    // pytorch is row major, so our outer loop should be channels

    // vel_x
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        file.write(reinterpret_cast<const char *>(&bins[y * width + x].vel.x), sizeof(float));
      }
    }
    // vel_y
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        file.write(reinterpret_cast<const char *>(&bins[y * width + x].vel.y), sizeof(float));
      }
    }
    // avg_vel
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        file.write(reinterpret_cast<const char *>(&bins[y * width + x].avg_vel), sizeof(float));
      }
    }
    // kinetic_energy
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        file.write(reinterpret_cast<const char *>(&bins[y * width + x].kinetic_energy), sizeof(float));
      }
    }
    // density
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        file.write(reinterpret_cast<const char *>(&bins[y * width + x].density), sizeof(float));
      }
    }
    // skip soil data for now
  }
};