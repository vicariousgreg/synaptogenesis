#ifdef __GUI__

#include "io/impl/saccade_module.h"

#include "saccade_window.h"

REGISTER_MODULE(SaccadeModule, "saccade");

SaccadeModule::SaccadeModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    this->window = SaccadeWindow::build(this);

    // Use input as default
    set_default_io_type(INPUT);

    for (auto layer : layers) {
        window->add_layer(layer, get_io_type(layer));
        auto layer_config = config->get_layer(layer);
        this->central[layer] = layer_config->get_bool("central", false);
    }
}

void SaccadeModule::feed_input_impl(Buffer *buffer) {
    window->lock();
    window->prepare_input_data();

    for (auto layer : layers) {
        if (get_io_type(layer) & INPUT) {
            if (this->central[layer])
                window->feed_central_input(layer, buffer->get_input(layer));
            else
                window->feed_input(layer, buffer->get_input(layer));
        }
    }
    window->unlock();
}

void SaccadeModule::report_output_impl(Buffer *buffer) {
    window->lock();
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            window->report_output(layer,
                buffer->get_output(layer),
                get_output_type(layer));
    window->unlock();
}

void SaccadeModule::report(Report* report) {
    int num_correct = 0;
    for (auto correct : correct_log)
        num_correct += correct;

    float avg_time = 0.0;
    for (auto time : time_log)
        avg_time += time;
    avg_time /= time_log.size();

    float std_dev = 0.0;
    for (auto time : time_log)
        std_dev += pow(time - avg_time, 2);
    std_dev = pow(std_dev / time_log.size(), 0.5);

    for (auto layer : layers) {
        report->add_report(this, layer,
            PropertyConfig({
                { "Total", std::to_string(correct_log.size()) },
                { "Correct", std::to_string(num_correct) },
                { "Average time", std::to_string(avg_time) },
                { "Standard deviation time", std::to_string(std_dev) },
            }));
    }
}

#endif
