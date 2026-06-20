// A small function exercising all three relations:
//   control flow : the for-loop (branches between blocks)
//   data flow    : sum / i carried across iterations (SSA phi + def-use)
//   memory order  : repeated loads from a[] (side-effecting memory reads)
int sum_array(const int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}