int count_positive(int* arr, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (arr[i] > 0) {
            count = count + 1;
        }
    }
    return count;
}
