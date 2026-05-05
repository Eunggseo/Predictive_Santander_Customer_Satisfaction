# Feature Importance & Business Insights - 分析交付物

Hazel 接手 Top 20 feature importance 后续分析的完整产出,基于你的 XGBoost FE 模型。

---

## 文件夹说明

```
Hazel_Feature_Analysis/
├── README.md                          ← 你正在看的这个文件
│
├── report/                            ← 正式报告(交付重点)
│   ├── feature_importance_business_insights_report.pdf   ← 5 页英文 PDF,直接发老师/stakeholder
│   ├── Feature_Importance_Business_Insights_EN.docx     ← 英文 Word 版,可编辑
│   └── Feature_Importance_Business_Insights_ZH.docx     ← 中文 Word 版,内部读起来快
│
├── shap/                              ← SHAP 分析
│   ├── generate_xgboost_fe_shap.py    ← 脚本
│   ├── xgboost_fe_top20_shap.csv      ← Top 20 SHAP 排名表
│   ├── xgboost_fe_top20_shap_summary.png    ← 蜂群图(presentation 主图)
│   ├── xgboost_fe_top20_shap_bar.png        ← Top 20 柱状图
│   └── xgboost_fe_shap_dependence_*.png (×6) ← Top 6 feature 的 dependence plots
│
└── 分析/                              ← 业务分桶分析
    ├── generate_xgboost_fe_business_analysis.py  ← 脚本
    ├── business_top_features_summary.csv         ← 4 个 feature 分桶汇总表
    ├── business_var15_age.png                    ← 年龄分桶 vs 不满意率
    ├── business_saldo_var30_binary.png           ← 余额(=0 vs >0)
    ├── business_saldo_var30_quintile.png         ← 余额(quintile)
    ├── business_ind_var30.png                    ← ind_var30 0/1
    └── business_ind_var26_cte.png                ← ind_var26_cte 0/1
```

---

## 三大核心结论

1. **SHAP 排名跟 gain 排名差异显著:** SHAP 把 var15(年龄)排第 1(占 30.7%),把 ind_var30 推到第 17 位。gain 容易高估二元 indicator,**presentation 建议用 SHAP 排名讲故事**。

2. **三大驱动因素 = 年龄 + 当前余额 + 历史余额**(SHAP 占比合计约 50%)。高风险客户画像 = **中年(30-60 岁)+ 低/零余额 + 低活跃度**。

3. **ind_var26_cte 是个被忽视的小子群:** 只占 2.7% 客户,但不满意率 6.68%,是平均的 1.7 倍。值得银行单独排查根因。

详细分析见 `report/` 里的报告。

---

## 怎么重跑

两个脚本都用了你的 `run_xgboost_fe.py` 接口(`from run_xgboost_fe import RNG_SEED, add_fe`),所以重跑步骤是:

1. 把 `run_xgboost_fe.py`、`train_clean.csv`、`test_clean.csv` 放在脚本同级目录
2. 安装 SHAP:`pip install shap`(business 脚本不需要)
3. 跑脚本:
   ```bash
   python generate_xgboost_fe_shap.py              # ~1-3 分钟
   python generate_xgboost_fe_business_analysis.py # 几秒
   ```

输出会写到当前目录。

---

## 注意事项

**关于 var15 < 23 桶:** 业务分桶图里 <23 那个桶不满意率是 0%,**不是因为年轻人都满意**,而是因为这部分人(5-22 岁)在原始数据里只有零散几十个/年龄,大概率是子账户/未成年/未激活账户。年轻主动客户的合理 baseline 是 23-30 桶(1.65%)。这个 caveat 在 PDF 报告 4.1 节有详细说明,讲 presentation 时一定要带上。
