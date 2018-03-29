#include <cmath>

#include "state/impl/sine_generator_attributes.h"
#include "util/error_manager.h"


REGISTER_ATTRIBUTES(SineGeneratorAttributes, "sine generator", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_RAND_ATTRIBUTE_KERNEL(SineGeneratorAttributes, Sine_generator_kernel,
    float *f_outputs = (float*)outputs;
    float wave = (1 + sin(att->frequency * att->iteration)) / 2;

    ,
    if (nid == 0) ++att->iteration;

    SHIFT_FLOAT_OUTPUTS(f_outputs, inputs[nid] * wave);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

SineGeneratorAttributes::SineGeneratorAttributes(Layer *layer)
        : Attributes(layer, FLOAT), iteration(0) {
    this->frequency = 2 * M_PI / 1000 // 2pi / 1000ms
        * std::stof(layer->get_parameter("frequency", "10"));
}
