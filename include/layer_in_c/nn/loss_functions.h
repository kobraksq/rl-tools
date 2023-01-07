#ifndef LAYER_IN_C_NN_LOSS_FUNCTIONS
#define LAYER_IN_C_NN_LOSS_FUNCTIONS

namespace layer_in_c::nn::loss_functions {
    template<typename T, int DIM, int BATCH_SIZE>
    T mse(const T a[DIM], const T b[DIM]) {
        T acc = 0;
        for(index_t i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            acc += diff * diff;
        }
        return acc / (DIM * BATCH_SIZE);
    }

    template<typename T, int DIM, int BATCH_SIZE>
    void d_mse_d_x(const T a[DIM], const T b[DIM], T d_a[DIM]) {
        for(index_t i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            d_a[i] = 2*diff/(DIM * BATCH_SIZE);
        }
    }
}


#endif