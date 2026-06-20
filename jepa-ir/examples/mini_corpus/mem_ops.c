// Exercises the memory-ordering relation: several stores/loads whose ORDER
// matters (the writes to *p happen in a defined sequence).
void scramble(int *p, int *q) {
    p[0] = 1;
    p[1] = p[0] + 2;
    q[0] = p[1];
    p[0] = q[0] * 3;
}