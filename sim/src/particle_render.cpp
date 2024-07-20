#include "particle_render.h"

// default point rendering
void render(const ParticleSystem &ps, SDL_Renderer *renderer, float render_scale)
{
  // batch render particles
  static std::vector<SDL_Point> points;
  points.clear();

  const auto &particles = ps.get_particles();
  for (const auto &particle : particles)
  {
    SDL_Point point = {
        static_cast<int>(particle.pos.x * render_scale),
        static_cast<int>(particle.pos.y * render_scale)};
    points.push_back(point);
    point.x++;
    points.push_back(point);
    point.y++;
    points.push_back(point);
    point.x--;
    points.push_back(point);
  }

  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  SDL_RenderDrawPoints(renderer, points.data(), points.size());
}

// color particles based on velocity
void render_velocity(const ParticleSystem &ps, SDL_Renderer *renderer, float render_scale)
{
  const auto &particles = ps.get_particles();
  for (const auto &particle : particles)
  {
    glm::vec2 vel = particle.vel * 0.025f;
    float vel_len = glm::length(vel);
    if (vel_len > 1.0f)
    {
      vel /= vel_len;
      vel_len = 1.0f;
    }
    SDL_SetRenderDrawColor(renderer,
                           255 * vel_len,
                           255 * (1.0f - vel_len),
                           0,
                           255);
    // use rectangles instead of points for better visibility
    SDL_Rect rect = {
        static_cast<int>(particle.pos.x * render_scale),
        static_cast<int>(particle.pos.y * render_scale),
        static_cast<int>(render_scale * 0.5f),
        static_cast<int>(render_scale * 0.5f)};
    SDL_RenderFillRect(renderer, &rect);
  }
}

void render_soil(const Soil &soil, SDL_Renderer *renderer, float render_scale)
{
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  for (const auto &cell : soil.get_grid())
  {
    for (const auto &particle : cell)
    {
      // render a circle at the radius
      glm::vec2 pos = particle.pos * render_scale;
      float radius = particle.radius * render_scale;
      // use a 16 sided polygon to approximate a circle
      glm::vec2 last_p = pos + glm::vec2{radius, 0.0f};
      for (int i = 0; i <= 16; ++i)
      {
        float angle = i / 16.0f * 2.0f * M_PI;
        glm::vec2 p = pos + glm::vec2{std::cos(angle), std::sin(angle)} * radius;
        SDL_RenderDrawLine(renderer, last_p.x, last_p.y, p.x, p.y);
        last_p = p;
      }
    }
  }

  // render grid
  SDL_SetRenderDrawColor(renderer, 127, 127, 127, 255);
  auto grid_width = soil.get_grid_width();
  auto grid_height = soil.get_grid_height();
  auto cell_size = soil.get_cell_size();
  for (int y = 0; y < grid_height; ++y)
  {
    for (int x = 0; x < grid_width; ++x)
    {
      SDL_Rect rect = {
          x * cell_size * render_scale,
          y * cell_size * render_scale,
          cell_size * render_scale,
          cell_size * render_scale};
      SDL_RenderDrawRect(renderer, &rect);
    }
  }
}

void render_bins(const Bins &bins, SDL_Renderer *renderer, float render_scale)
{
  if (bins.bins.empty())
  {
    return;
  }
  for (int y = 0; y < bins.height; ++y)
  {
    for (int x = 0; x < bins.width; ++x)
    {
      const auto &bin = bins.bins[y * bins.width + x];
      float normalized_density = bin.density / (2 * TARGET_PRESSURE);

      float r = normalized_density;
      float g = normalized_density;
      float b = normalized_density;

      // color based on velocity
      auto vel = bin.vel * 0.025f;
      float vel_len = glm::length(vel);
      if (vel_len > 1.0f)
      {
        vel /= vel_len;
        vel_len = 1.0f;
      }
      float vel_r = 0.5f + 0.5f * vel.x;
      float vel_g = 0.5f + 0.5f * vel.y;
      r *= vel_r;
      g *= vel_g;

      // just use grey scale for now
      SDL_SetRenderDrawColor(renderer,
                             255 * r,
                             255 * g,
                             255 * b,
                             255);
      SDL_Rect rect = {
          static_cast<int>((bins.start.x + x * bins.bin_size) * render_scale),
          static_cast<int>((bins.start.y + y * bins.bin_size) * render_scale),
          static_cast<int>(bins.bin_size * render_scale),
          static_cast<int>(bins.bin_size * render_scale)};
      SDL_RenderFillRect(renderer, &rect);
    }
  }
}