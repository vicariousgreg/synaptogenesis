#include "driver.h"
#include "izhikevich_driver.h"
#include "rate_encoding_driver.h"

Driver* build_driver(Model* model) {
    Driver* driver;
    if (model->driver_string == "izhikevich") {
        driver = new IzhikevichDriver();
    } else if (model->driver_string == "rate_encoding") {
        driver = new RateEncodingDriver();
    } else {
        throw "Unrecognized driver type!";
    }
    driver->state->build(model);
    driver->model = model;
    return driver;
}

