原版的：mat/algorithms/mat/algorithm/ma_transformer_origin.py

完全遵循Titans理论、支持step-by-step递归式长期记忆更新，持久记忆与长期记忆模块集成在MAT编码器中的最新版ma_transformer.py骨架。
代码既包含你需要的**“持久记忆token”，也包含“长期记忆模块”**（严格实现公式13/14，并支持数据自适应更新参数），你只需要补全decoder和action部分即可直接用在你的MARL项目。
mat/algorithms/mat/algorithm/ma_transformer.py