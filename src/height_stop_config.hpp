#ifndef HEIGHT_STOP_CONFIG_HPP
#define HEIGHT_STOP_CONFIG_HPP

// Temporary convergence test: grow until height N*L instead of L.
// Restore this value to 1 to return to the standard stopping height.
constexpr int HEIGHT_STOP_MULTIPLIER = 1;

#endif // HEIGHT_STOP_CONFIG_HPP
