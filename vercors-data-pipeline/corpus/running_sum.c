void running_sum(int* arr, int len) {
    for (int i = 1; i < len; i++) {
        arr[i] = arr[i] + arr[i - 1];
    }
}
