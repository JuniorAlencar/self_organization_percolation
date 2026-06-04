#ifndef HEIGHT_STOP_CONFIG_HPP
#define HEIGHT_STOP_CONFIG_HPP

constexpr int HEIGHT_STOP_MULTIPLIER = 1;

struct GrowthStopConfig {
    int height_multiplier = HEIGHT_STOP_MULTIPLIER;
    int height_extra_layers = 0;
    bool dynamic_height = false;
    bool stop_at_percolation = true;
    bool stop_at_equilibrium = false;
    int equilibrium_consecutive_steps = 10;
    double equilibrium_rel_tol = 2.5e-2;
    double equilibrium_abs_tol = 1.0e-6;
    int hard_max_steps = -1;
    int dynamics_window_steps = 300;
    double min_rel_error_improvement = 5.0e-2;
    double derivative_abs_tol = 5.0e-4;
    double control_derivative_abs_tol = 5.0e-4;
    double derivative_sign_change_fraction = 0.35;
};

constexpr int GROWTH_TEST_DYNAMICS_WINDOW_FACTOR = 30;

#endif // HEIGHT_STOP_CONFIG_HPP
