// #include "MultiRateBase.h"
#include "utils/units.h"

namespace kintera
{


class Reaction;
class ReactionRate
{
public:
    ReactionRate() {}
    virtual ~ReactionRate() = default;
    // Copy assignment operator and copy constructor are deleted

    virtual const string type() const = 0;

    //! String identifying sub-type of reaction rate specialization
    virtual const string subType() const {
        return "";
    }

    virtual void setParameters(const AnyMap& node) = 0;
    virtual void setParameters(const AnyValue& rate) = 0;

    virtual torch::Tensor evalRate(torch::Tensor T, torch::Tensor P) const = 0;
};

}

#endif