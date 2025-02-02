// kintera
#include "kin7_xsection.hpp"

#include <math/interpolation.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

namespace kintera {

Kin7XsectionImpl::Kin7XsectionImpl(S8RTOptions const& options_)
    : options(options_) {
  reset();
}

void Kin7XsectionImpl::reset() {
  auto full_path = find_resource(options.opacity_file());

  FILE* file = fopen(full_path.c_str(), "r");

  TORCH_CHECK(file, "Could not open file: ", filename);

  std::vector<double> wavelength;
  std::vector<double> xsection;

  // first cross section data is always the photoabsorption cross section (no
  // dissociation)
  int nbranch = branches.size();
  int min_is = 9999, max_ie = 0;

  char* line = NULL;
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
    int num =
        sscanf(line, "%60c%4d%4d%4d%6f", equation, &is, &ie, &nwave, &temp);
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
      for (int i = 0; i < nrows; i++) getline(&line, &len, file);
    } else {
      for (int i = 0; i < nrows; i++) {
        getline(&line, &len, file);

        for (int j = 0; j < ncols; j++) {
          float wave, cross;
          int num = sscanf(line + 17 * j, "%7f%10f", &wave, &cross);
          TORCH_CHECK(num == 2, "Cross-section format from file '", filename,
                      "' is wrong.");
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
  kdata = register_buffer(
      "kdata", torch::tensor(xsection).view({wavelength.size(), nbranch}));
}

torch::Tensor Kin7XsectionImpl::forward(torch::Tensor wave, torch::Tensor aflux,
                                        torch::optional<torch::Tensor> kcross,
                                        torch::optional<torch::Tensor> temp) {
  int nwave = aflux.size(0);
  int ncol = aflux.size(0);
  int nlyr = aflux.size(1);
  int nbranch = options.branches().size();
  int nspecies = options.species().size();

  auto dims = torch::tensor(
      {kwave.size(0)},
      torch::TensorOptions().dtype(torch::kInt64).device(wave.device()));

  auto out = torch::empty({nwave, ncol, nlyr, nbranch},
                          torch::TensorOptions().device(wave.device()));

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(out.sizes())
          .add_output(out)
          .add_owned_const_input(
              wave.view({-1, 1, 1, 1}).expand({-1, ncol, nlyr, nbranch}))
          .build();

  if (wave.is_cpu()) {
    call_interpn_cpu<MAX_PHOTO_BRANCHES>(iter, kdata, kwave, dims, /*ndim=*/1);
  } else if (wave.is_cuda()) {
    // call_interpn_cuda<MAX_PHOTO_BRANCHES>(iter, kdata, kwave, dims, 1);
  } else {
    TORCH_CHECK(false, "Unsupported device");
  }

  // save the interpolated cross section
  // (nreaction, nwave, ncol, nlyr)
  if (kcross.has_value()) {
    kcross.value()[options.reaction_id()].copy_(out.sum(3));
  }

  // (ncol, nlyr, nbranch)
  auto rate = torch::trapezoid(aflux.unsqueeze(-1) * out, wave, 0);
  // total_rate = rate.sum(2);

  // (ncol, nlyr, nspecies)
  return (rate.unsqueeze(-1) * stoich.view({1, 1, nbranch, nspecies)).sum(2)
    / rate.sum(2).unsqueeze(-1);
}

}  // namespace kintera
