void f(int *p, int a) {
    int x = a + 1;
    p[0] = x;
    if (a > 0)
        p[1] = x;
}
