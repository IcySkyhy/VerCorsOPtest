int clamp_value(int x, int lo, int hi) {
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    } else {
        return x;
    }
}
