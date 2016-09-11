#include <iostream>
#include <vector>
#include "environment.h"

int main(void) {
    Environment env;
    int size = 60;
    int iterations = 250;

    double a = 0.02;
    double b = 0.25;
    double c = -65;
    double d = 0.05;

    int pos = env.add_layer(size, 1, a, b, c, d);
    int neg = env.add_layer(size, -1, a, b, c, d);
    env.connect_layers(pos, pos, true);
    env.connect_layers(pos, neg, true);
    env.connect_layers(neg, pos, true);
    env.connect_layers(neg, neg, true);

    for (int i = 0 ; i < iterations ; ++i) {
        //cout << "Currents: " << env.currents[0] << " " << env.currents[1] << "\n";
        //cout << "Voltages: " << env.voltages[0] << " " << env.voltages[1] << "\n";
        //cout << "Spikes: " << env.spikes[0] << " " << env.spikes[1] << "\n\n";
        for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
            char c = env.spikes[nid] ? '.' : ' ';
            //cout << env.spikes[nid];
            cout << c;
        }
        cout << "|\n";
        env.cycle();
    }

    /*
    env.add_neuron(0,0,0,0);
    env.voltages[0] = 100;
    cout << env.voltages[0] << "\n";

    env.cycle();
    cout << env.voltages[0] << "\n";
    cout << env.spikes[0] << "\n";

    env.voltages[0] = 0;
    env.cycle();
    cout << env.spikes[0] << "\n";
    */
    return 0;
}
