#pragma once

namespace kinetera {

//! \brief stores interpolated photo x-sections for all photolysis reactions
struct XsectionImpl {
  //! interpoalted photo x-section [cm^2 molecule^-1] at common wavelengths
  //! (nwave, ncol, nlyr, MAX_PHOTO_BRANCHES)
  torch::Tensor kcross;
};

using Xsection = std::shared_ptr<XsectinoImpl>;

}  // namespace kinetera
