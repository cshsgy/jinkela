// kintera
#include "kin7_xsection.hpp"

#include <math/interpolation.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

namespace kintera {

Kin7XsectionImpl::Kin7XsectionImpl(S8RTOptions const& options_) 
  : options(options_) 
{
  reset();
}

void Kin7XsectionImpl::reset() {
  auto full_path = find_resource(options.opacity_file());

  FILE* file = fopen(full_path.c_str(), "r");

  TORCH_CHECK(file, "Could not open file: ", filename);

  std::vector<double> wavelength;
  std::vector<double> xsection;

  // first cross section data is always the photoabsorption cross section (no dissociation)
  int nbranch = branches.size();
  int min_is = 9999, max_ie = 0;

  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  // Read each line from the file
  while ((read = getline(&line, &len, file)) != -1) {
    // Skip empty lines or lines containing only whitespace
    if (line[0] == '\n') continue;

    char equation[61];
    int is, ie, nwave;
    float temp;

    // read header
    int num = sscanf(line, "%60c%4d%4d%4d%6f", equation, &is, &ie, &nwave, &temp);
    min_is = std::min(min_is, is);
    max_ie = std::max(max_ie, ie);

    TORCH_CHECK(num == 5, "Header format from file '", filename, "' is wrong.");
    // initialize wavelength and xsection for the first time
    if (wavelength.size() == 0) {
      wavelength.resize(nwave);
      xsection.resize(nwave * nbranch);
    }

    // read content
    int ncols = 7;
    int nrows = ceil(1. * nwave / ncols);
    
    equation[60] = '\0';
    auto product = parseCompString(equation);

    auto it = std::find(branches.begin(), branches.end(), product);

    if (it == branches.end()) {
      // skip this section
      for (int i = 0; i < nrows; i++)
        getline(&line, &len, file);
    } else {
      for (int i = 0; i < nrows; i++) {
        getline(&line, &len, file);

        for (int j = 0; j < ncols; j++) {
          float wave, cross;
          int num = sscanf(line + 17*j, "%7f%10f", &wave, &cross);
          TORCH_CHECK(num == 2, "Cross-section format from file '", filename, "' is wrong.");
          int b = it - branches.begin();
          int k = i * ncols + j;

          if (k >= nwave) break;
          // Angstrom -> nm
          wavelength[k] = wave * 10.;
          // cm^2
          xsection[k * nbranch + b] = cross;
        }
      }
    }
  }

  // remove unused wavelength and xsection
  wavelength = std::vector<double>(wavelength.begin() + min_is - 1,
                                   wavelength.begin() + max_ie);

  xsection = std::vector<double>(xsection.begin() + (min_is - 1) * nbranch,
                                 xsection.begin() + max_ie * nbranch);

  // A -> A is the total cross section in kinetics7 format
  // need to subtract the other branches

  for (size_t i = 0; i < wavelength.size(); i++) {
    for (int j = 1; j < nbranch; j++) {
      xsection[i * nbranch] -= xsection[i * nbranch + j];
    }
    xsection[i * nbranch] = std::max(xsection[i * nbranch], 0.);
  }

  free(line);
  fclose(file);

  kwave = register_buffer("kwave", torch::tensor(wavelength));
  kdata = register_buffer("kdata", torch::tensor(xsection).view({wavelength.size(), nbranch}));
}

torch::Tensor Kin7XsectionImpl::forward(torch::Tensor wave, 
                                        torch::Tensor aflux,
                                        torch::optional<torch::Tensor> temp) {
  int nwve = wave.size(0);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);
  constexpr int nprop = 2 + S8RTOptions::npmom;

  auto out = torch::zeros({nwve, ncol, nlyr, nprop}, conc.options());
  auto dims = torch::tensor(
      {kwave.size(0)},
      torch::TensorOptions().dtype(torch::kInt64).device(conc.device()));

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/3)
                  .add_output(out)
                  .add_owned_const_input(wave.view({-1, 1, 1, 1}))
                  .build();

  if (conc.is_cpu()) {
    call_interpn_cpu<nprop>(iter, kdata, kwave, dims, 1);
  } else if (conc.is_cuda()) {
    // call_interpn_cuda<nprop>(iter, kdata, kwave, dims, 1);
  } else {
    TORCH_CHECK(false, "Unsupported device");
  }

  // attenuation [1/m]
  out.select(3, 0) *= conc.select(2, options.species_id()).unsqueeze(0);

  // attenuation weighted single scattering albedo [1/m]
  out.select(3, 1) *= out.select(3, 0);

  return out;
}

}  // namespace kintera
