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
    SDL_SetRenderDrawColor(renderer, 255 * vel_len, 255 * (1.0f - vel_len), 0, 255);
    // use rectangles instead of points for better visibility
    SDL_Rect rect = {
        static_cast<int>(particle.pos.x * render_scale),
        static_cast<int>(particle.pos.y * render_scale),
        static_cast<int>(render_scale * 0.5f),
        static_cast<int>(render_scale * 0.5f)};
    SDL_RenderFillRect(renderer, &rect);
  }
}