void array_add_scalar(int* arr, int len, int scalar) {
    for (int i = 0; i < len; i++) {
        arr[i] = arr[i] + scalar;
    }
}
