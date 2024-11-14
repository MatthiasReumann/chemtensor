#pragma once

#include "mpo.h"
#include "aligned_memory.h"

struct gmap
{
    struct mpo **data;    // List of MPO pairs accessible like an dictionary with tuple key via get_value(...)
    long N;                 // THC Rank `N`. Length of `data` is `2N`
};

void allocate_gmap(struct gmap *g, const long N);

void get_gmap_pair(const struct gmap *g, const int i, const int s, struct mpo **pair);