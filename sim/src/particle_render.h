#pragma once

#include "SDL.h"
#include "particle_system.h"

void render(const ParticleSystem &ps, SDL_Renderer *renderer, float render_scale);