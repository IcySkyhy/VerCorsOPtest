/*@
  requires x >= 0;          // 前置条件：输入参数必须大于等于0
  ensures \result > 0;      // 后置条件：返回值必须严格大于0
@*/
int add_one(int x) {
    return x + 1;
}