#define DIMENSIONS 30
#define N 6000
#define THREADS 256
#define DIMENSIONS_LOWER 2
#define MAX_LEARNING_RATE 200000
#define P_MULTIPLIER 4.0
#define TILE_WIDTH 32
#define ITERATIONS 1000

#define DIMENSION_LIMIT_FOR_USING_TILED_KERNEL 5 // if number of dimensions is less than this value, we use the non-tiled kernel