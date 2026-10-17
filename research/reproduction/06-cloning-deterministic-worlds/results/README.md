# 运行结果（待 GPU 真实训练）

**当前状态：尚未执行 Maze9 的 VAE / GRWM 训练与扩散世界模型评价。** 本目录只存放将来通过官方流程实际生成的实验指标与结果图，不放从论文截图仿造的曲线。

官方运行完成后，在项目根目录执行：

```bash
python code/export_results.py
```

期望导出 `maze9_vae_vs_grwm_norm_mse_per_step.png/.pdf/.csv`、两种方法各自的 `*_rollout_pixel_mse_summary.json`、汇总表。图中横轴是未来预测步数，纵轴是归一化像素 MSE，越低越好。

正式记录还需要补充：显卡、显存、数据版本、固定 Git commit、训练配置（debug/paper）、随机种子、训练完成日志和曲线。不将 debug 数值称为论文主结果。
