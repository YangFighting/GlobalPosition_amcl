# GlobalPosition_amcl

相对于move_base 中的AMCL改变如下
1. 删除了对TF树的广播,即不会对TF树产生变化,便于将数据导出
2. 将 地图数据,激光雷达数据, 里程计数据 到处成txt

