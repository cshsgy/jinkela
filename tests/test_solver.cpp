#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetic_rate.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/rate_constant.hpp>
#include <kintera/kinetics/species_rate.hpp>
#include <kintera/kinetics/reaction_system.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>
#include <kintera/solver/explicit.hpp>
#include <kintera/solver/implicit.hpp>
#include <kintera/utils/parse_yaml.hpp>
#include <kintera/utils/stoichiometry.hpp>

void run_solver_test(const std::string& solver_type, 
                    kintera::ReactionSystem& reaction_system,
                    torch::Tensor& C,
                    const torch::Tensor& P,
                    const torch::Tensor& Temp,
                    double dt,
                    int n_steps) {
    
    std::cout << "\nRunning " << solver_type << " solver test:\n";
    std::cout << "Initial concentrations:\n" << C << "\n\n";

    if (solver_type == "explicit") {
        kintera::ExplicitSolver solver;
        for (int step = 0; step < n_steps; ++step) {
            solver.time_march(C, P, Temp, dt, reaction_system);
            std::cout << "Step " << step + 1 << " concentrations:\n" << C << "\n\n";
        }
    } else {
        kintera::ImplicitSolver solver;
        for (int step = 0; step < n_steps; ++step) {
            solver.time_march(C, P, Temp, dt, reaction_system);
            std::cout << "Step " << step + 1 << " concentrations:\n" << C << "\n\n";
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
            return 1;
        }
        std::filesystem::path yaml_file = argv[1];

        // Set up rate constant
        auto rop = kintera::RateConstantOptions();
        rop.types({"Arrhenius"});
        rop.reaction_file(yaml_file.string());
        auto rate_constant = kintera::RateConstant(rop);

        // Set up kinetic rate
        auto kop = kintera::KineticRateOptions();
        kop.reactions() = kintera::parse_reactions_yaml(rop.reaction_file(), rop.types());
        std::cout << "Successfully parsed " << kop.reactions().size() << " reactions\n\n";

        // Collect all unique species
        std::set<std::string> species_set;
        for (const auto& reaction : kop.reactions()) {
            for (const auto& [species, _] : reaction.reactants()) {
                species_set.insert(species);
            }
            for (const auto& [species, _] : reaction.products()) {
                species_set.insert(species);
            }
        }
        kop.species() = std::vector<std::string>(species_set.begin(), species_set.end());

        // Create kinetics rates module
        auto kinetics = kintera::KineticRate(kop);
        kinetics->to(torch::kFloat64);

        // Create test conditions
        int64_t ncol = 2;
        int64_t nlyr = 3;
        int64_t nspecies = static_cast<int64_t>(kop.species().size());

        auto Temp = 300. * torch::ones({ncol, nlyr}, torch::kFloat64);
        auto P = torch::ones({ncol, nlyr}, torch::kFloat64) * 101325.;
        auto C = 1e-3 * torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);

        // Create reaction system
        kintera::ReactionSystem reaction_system(kinetics, rate_constant);

        // Test parameters
        double dt = 0.1;
        int n_steps = 5;

        // Run explicit solver test
        auto C_explicit = C.clone();
        run_solver_test("explicit", reaction_system, C_explicit, P, Temp, dt, n_steps);

        // Run implicit solver test
        auto C_implicit = C.clone();
        run_solver_test("implicit", reaction_system, C_implicit, P, Temp, dt, n_steps);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
