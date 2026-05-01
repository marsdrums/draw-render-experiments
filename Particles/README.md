# Particle renderer with visible-particle radix sort

This repository is your original patch/package plus a front-to-back radix sort over the compacted visible particle index list.

## What changed

- Added GPU depth-key generation for visible particles
- Added 4-pass 8-bit radix sort on `buff_visible_indices`
- Sorting happens after frustum culling + compaction and before drawing
- `count(N)` now resizes buffers and dispatches dynamically for any particle count
- `count(0)` is also handled safely

## New shader files

- `comp_build_sort_keys.comp`
- `comp_radix_clear.comp`
- `comp_radix_histogram.comp`
- `comp_radix_scan.comp`
- `comp_radix_scatter.comp`

## Files modified

- `basic_particle_rendering.js`

## Integration note

The sorted output remains in `buff_visible_indices`, so your existing draw shader and indirect draw path continue to work without changing `draw_particles.rend`.
