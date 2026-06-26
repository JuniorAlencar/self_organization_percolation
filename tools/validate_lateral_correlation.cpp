#include "LateralCorrelation.hpp"

#include <iostream>
#include <string>

int main() {
    std::string error;
    if (!validate_lateral_observables(&error)) {
        std::cerr << "Validation failed: " << error << '\n';
        return 1;
    }

    std::cout << "Lateral correlation validation passed." << std::endl;
    return 0;
}
