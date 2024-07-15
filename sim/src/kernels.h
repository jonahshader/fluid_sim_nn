#pragma once

#include <functional>
#include <cmath>

#include "custom_math.h"

constexpr float smoothstep_kernel_volume(float kernel_radius)
{
  return 3.0f * M_PI_F * kernel_radius * kernel_radius / 10.0f;
}

/**
 * @brief Creates a smoothstep kernel function with a given kernel radius s.
 * The smoothstep kernel function takes in radius r and has a global maximum at r=0,
 * and smoothly decreases to 0 as r approaches s. The volume of the kernel is normalized to 1.
 * See https://en.wikipedia.org/wiki/Smoothstep
 *
 * @param s Kernel radius
 * @return std::function<float(float)> Smoothstep kernel function
 */
std::function<float(float)> make_smoothstep_kernel(float s)
{
  float kernel_vol_inv = 1.0f / smoothstep_kernel_volume(s);
  return [=](float r)
  {
    if (r >= s)
      return 0.0f;
    float q = 1 - r / s;
    return q * q * (3.0f - 2.0f * q) * kernel_vol_inv;
  };
}

/**
 * @brief Creates a smoothstep kernel derivative function with a given kernel radius s.
 * The smoothstep kernel derivative function takes in radius r and has a global minimum at r=0,
 * and smoothly increases to 0 as r approaches s.
 * See https://en.wikipedia.org/wiki/Smoothstep
 *
 * @param s Kernel radius
 * @return std::function<float(float)> Smoothstep kernel derivative function
 */
std::function<float(float)> make_smoothstep_kernel_derivative(float s)
{
  float kernel_vol_inv = 1.0f / smoothstep_kernel_volume(s);
  return [=](float r)
  {
    if (r >= s)
      return 0.0f;
    float q = 1 - r / s;
    return -6.0f * q * (1 - q) * kernel_vol_inv;
  };
}

std::function<float(float)> make_sharp_kernel(float s)
{
  float volume_inv = 6.0f / (M_PI_F * std::pow(s, 4));
  return [=](float r)
  {
    if (r >= s)
      return 0.0f;
    float q = s - r;
    return q * q * volume_inv;
  };
}

std::function<float(float)> make_sharp_kernel_derivative(float s)
{
  float volume_inv = 6.0f / (M_PI_F * std::pow(s, 4));
  return [=](float r)
  {
    if (r >= s)
      return 0.0f;
    return (-2 * s + 2 * r) * volume_inv;
  };
}