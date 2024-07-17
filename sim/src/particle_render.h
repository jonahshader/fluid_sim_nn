#pragma once

#include "SDL.h"
#include "particle_system.h"

// default point rendering
void render(const ParticleSystem &ps, SDL_Renderer *renderer, float render_scale);
// color particles based on velocity
void render_velocity(const ParticleSystem &ps, SDL_Renderer *renderer, float render_scale);