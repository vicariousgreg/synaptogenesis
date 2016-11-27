// Update weight
float old_weight = conn_data.weights[index];
float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
- (WEIGHT_DECAY * (old_weight - baseline[index]));
conn_data.weights[index] = (new_weight > conn_data.max_weight) ? conn_data.max_weight : new_weight;

// Update weight
float old_weight = conn_data.weights[index];
float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
- (WEIGHT_DECAY * (old_weight - baseline[index]));
conn_data.weights[index] = (new_weight > conn_data.max_weight) ? conn_data.max_weight : new_weight;

// Update weight
float sum = conn_data.inputs[index];
float old_weight = conn_data.weights[index];
float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
- (WEIGHT_DECAY * (old_weight - baseline[index]));
conn_data.weights[index] = (new_weight > conn_data.max_weight) ? conn_data.max_weight : new_weight;


// Update weight
int index = weight_index;
float old_weight = conn_data.weights[index];
float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
- (WEIGHT_DECAY * (old_weight - baseline[index]));
conn_data.weights[index] = (new_weight > conn_data.max_weight) ? conn_data.max_weight : new_weight;
