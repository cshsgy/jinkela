#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <kintera/solver/explicit.h>

int main() {
    try {
        int64_t ncol = 2;
        int64_t nlyr = 3;
        int64_t nspecies = 3;

        auto C = torch::zeros({ncol, nlyr, nspecies}, torch::kFloat64);
        C.select(-1, 0) = 1e-3;
        C.select(-1, 1) = 2e-3;
        C.select(-1, 2) = 3e-3;
        
        auto stoich_matrix = torch::tensor({
            {-1.0, 0.0, 1.0},  // A
            {1.0, -1.0, 0.0},  // B
            {0.0, 1.0, -1.0}    // C
        }, torch::kFloat64);

        auto rates = torch::zeros({ncol, nlyr, 3}, torch::kFloat64);
        rates.select(-1, 0) = 1e-3;
        rates.select(-1, 1) = 1e-3;
        rates.select(-1, 2) = 1e-3;

        double dt = 0.1;
        double max_rel_change = 0.1;
        int n_steps = 5;

        std::cout << "Initial concentrations:\n" << C << "\n\n";

        for (int step = 0; step < n_steps; ++step) {
            kintera::explicit_solve(C, rates, stoich_matrix, dt, max_rel_change);
            
            std::cout << "Step " << step + 1 << " concentrations:\n";
            std::cout << "Species A: " << C.select(-1, 0) << "\n";
            std::cout << "Species B: " << C.select(-1, 1) << "\n";
            std::cout << "Species C: " << C.select(-1, 2) << "\n\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
