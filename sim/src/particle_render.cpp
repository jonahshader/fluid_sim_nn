#include "particle_render.h"

void render(const ParticleSystem &ps, SDL_Renderer *renderer)
{
  // batch render particles
  static std::vector<SDL_Point> points;
  points.clear();

  const auto &particles = ps.get_particles();
  for (const auto &particle : particles)
  {
    SDL_Point point = {static_cast<int>(particle.pos.x), static_cast<int>(particle.pos.y)};
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