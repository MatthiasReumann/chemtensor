#include "gmap.h"

void allocate_gmap(struct gmap *g, const long N)
{
    assert(N > 0);
    g->N = N;
    g->data = ct_malloc(2 * N * sizeof(struct mpo*));
    for(size_t i = 0; i < 2 * N; i++) {
        g->data[i] = ct_malloc(2 * sizeof(struct mpo));
    }
}


void get_gmap_pair(const struct gmap *g, const int i, const int s, struct mpo **pair)
{
    assert(i + g->N * s < 2 * g->N);
    *(pair) = g->data[i + g->N * s];
}