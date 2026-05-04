/*@
  requires len >= 0;
  requires arr != NULL;
  // 分离逻辑核心：声明对数组 0 到 len-1 索引的所有元素，拥有 1/2 的权限（代表只读权限）
  requires (\forall* int j; 0 <= j && j < len; Perm(arr[j], 1/2));
  ensures \result >= 0;
@*/
int sum_positive(int* arr, int len) {
    int sum = 0;
    
    /*@
      loop_invariant 0 <= i && i <= len;
      loop_invariant sum >= 0;
      // 循环中必须保持这些权限不丢失
      loop_invariant (\forall* int j; 0 <= j && j < len; Perm(arr[j], 1/2));
    @*/
    for (int i = 0; i < len; i++) {
        if (arr[i] > 0) {
            sum += arr[i];
        }
    }
    return sum;
}