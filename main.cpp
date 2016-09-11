#include <iostream>
#include <vector>
#include "environment.h"

int main(void) {
    Environment env;
    int size = 100;
    int iterations = 250;

    int lid = env.add_layer(size, 1, 0.02, 0.25, -65, 0.05);
    env.connect_layers(lid, lid, true);

    for (int i = 0 ; i < iterations ; ++i) {
        //cout << "Currents: " << env.currents[0] << " " << env.currents[1] << "\n";
        //cout << "Voltages: " << env.voltages[0] << " " << env.voltages[1] << "\n";
        //cout << "Spikes: " << env.spikes[0] << " " << env.spikes[1] << "\n\n";
        for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
            char c = env.spikes[nid] ? '.' : ' ';
            //cout << env.spikes[nid];
            cout << c;
        }
        cout << "\n";
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
