#ifndef model_builder_h
#define model_builder_h

#include "model/model.h"

Model* load_model(std::string path);
void save_model(Model *model, std::string path);

#endif
