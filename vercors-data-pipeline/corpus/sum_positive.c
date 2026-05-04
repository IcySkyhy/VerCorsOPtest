int sum_positive(int* arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        if (arr[i] > 0) {
            sum += arr[i];
        }
    }
    return sum;
}
