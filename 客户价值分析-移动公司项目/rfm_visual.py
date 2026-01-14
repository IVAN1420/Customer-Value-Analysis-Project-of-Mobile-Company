# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np

# 忽略无关警告，运行无冗余报错
warnings.filterwarnings('ignore')

# ===================== 全局配置：中文正常显示 + 负号正常显示 【彻底解决乱码】 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13
plt.rcParams['figure.figsize'] = (12, 8)  # 每张独立图的尺寸，高清展示

# ===================== 精准匹配你的文件路径【无需修改】 =====================
file_path = r"D:\桌面\数据挖掘\pythonProject1\.venv\rfm_customer_cluster.csv"
df = pd.read_csv(file_path)

# ===================== 控制台打印数据信息 + RFM核心业务指标 =====================
print("✅ 数据读取成功！✅")
print(f"📊 总共分析用户数：{len(df):,} 人")
print("🔍 数据前5行预览：")
print(df.head())
print("="*70)
print("📈 RFM 用户价值分析核心指标")
print("="*70)
total_user = len(df)
avg_age = df['age'].mean()
male_rate = df['gender'].value_counts()[1]/total_user*100
female_rate = df['gender'].value_counts()[2]/total_user*100
repeat_rate = df['is_repeat_customer'].sum()/total_user*100
avg_money = df['M'].mean()
high_value_money = df[df['customer_value_level']=='高价值客户']['M'].mean()

print(f"总用户数        ：{total_user:,} 人")
print(f"用户平均年龄    ：{avg_age:.1f} 岁")
print(f"男性用户占比    ：{male_rate:.2f}%")
print(f"女性用户占比    ：{female_rate:.2f}%")
print(f"整体复购率      ：{repeat_rate:.2f}%")
print(f"整体月均消费金额：{avg_money:.2f} 元")
print(f"高价值客户月均消费：{high_value_money:.2f} 元")
print("="*70)

# ===================== 核心：生成8张独立图片 + 分别保存为8个文件【全部独立、无拼接】 =====================
print("\n📌 开始生成可视化图片，每张图将独立保存为文件...")

# ---------- 图片1：客户价值等级分布 ----------
plt.figure()
val_cnt = df['customer_value_level'].value_counts()
colors1 = ['#e74c3c', '#f39c12', '#f1c40f']
plt.pie(val_cnt.values, labels=val_cnt.index, autopct='%1.2f%%', colors=colors1, shadow=True)
plt.title('💎 用户客户价值等级分布', fontsize=18, fontweight='bold', pad=20)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\1_客户价值等级分布.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：1_客户价值等级分布.png")

# ---------- 图片2：用户聚类标签分布 ----------
plt.figure()
clus_cnt = df['cluster_cn_label'].value_counts()
colors2 = ['#27ae60', '#3498db', '#9b59b6', '#e67e22']
bars = plt.bar(clus_cnt.index, clus_cnt.values, color=colors2, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('🏆 用户聚类标签分布', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('用户数量（人）', fontsize=14)
plt.xticks(rotation=8)
for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, f'{int(bar.get_height())}', ha='center', fontsize=12)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\2_用户聚类标签分布.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：2_用户聚类标签分布.png")

# ---------- 图片3：客户复购率分布 ----------
plt.figure()
rep_cnt = df['is_repeat_customer'].map({1:'复购客户',0:'非复购客户'}).value_counts()
colors3 = ['#1abc9c', '#e74c3c']
plt.pie(rep_cnt.values, labels=rep_cnt.index, autopct='%1.2f%%', colors=colors3, shadow=True)
plt.title('🛒 用户复购率分布', fontsize=18, fontweight='bold', pad=20)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\3_客户复购率分布.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：3_客户复购率分布.png")

# ---------- 图片4：用户年龄分布 ----------
plt.figure()
plt.hist(df['age'], bins=15, color='#34495e', alpha=0.7, edgecolor='black', linewidth=1)
plt.title('👨👩 用户年龄分布', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('年龄', fontsize=14)
plt.ylabel('用户数量（人）', fontsize=14)
plt.axvline(avg_age, color='red', linestyle='--', label=f'平均年龄 {avg_age:.1f}岁', linewidth=2)
plt.legend(loc='upper right', fontsize=12)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\4_用户年龄分布.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：4_用户年龄分布.png")

# ---------- 图片5：用户性别分布 ----------
plt.figure()
gen_cnt = df['gender'].map({1:'男性',2:'女性'}).value_counts()
colors5 = ['#2c3e50', '#e74c3c']
bars5 = plt.bar(gen_cnt.index, gen_cnt.values, color=colors5, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('👫 用户性别分布', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('用户数量（人）', fontsize=14)
for bar in bars5:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, f'{int(bar.get_height())}', ha='center', fontsize=12)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\5_用户性别分布.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：5_用户性别分布.png")

# ---------- 图片6：各价值等级平均月消费金额 ----------
plt.figure()
m_avg = df.groupby('customer_value_level')['M'].mean().sort_values(ascending=False)
colors6 = ['#e74c3c', '#f39c12', '#f1c40f']
bars6 = plt.bar(m_avg.index, m_avg.values, color=colors6, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('💰 各价值等级平均月消费金额', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('平均消费金额（元）', fontsize=14)
for bar in bars6:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.2f}', ha='center', fontsize=12)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\6_各价值等级平均消费.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：6_各价值等级平均消费.png")

# ---------- ✅ 新增图片7：聚类结果统计柱状图【重点新增】----------
plt.figure()
cluster_stat = df['cluster_label'].value_counts().sort_index()
colors7 = ['#6c5ce7', '#fd79a8', '#fdcb6e', '#00b894']
bars7 = plt.bar([f'聚类_{i}' for i in cluster_stat.index], cluster_stat.values, color=colors7, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('📊 聚类结果数量统计柱状图', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('聚类标签', fontsize=14)
plt.ylabel('用户数量（人）', fontsize=14)
for bar in bars7:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, f'{int(bar.get_height())}', ha='center', fontsize=12)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\7_聚类结果数量统计.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：7_聚类结果数量统计.png")

# ---------- ✅ 新增图片8：客户群特征雷达图【重点新增、核心分析】----------
plt.figure(figsize=(10, 10))
# 按客户价值等级计算核心特征指标
value_levels = ['高价值客户', '中价值客户', '一般价值客户']
metrics = ['月均消费金额', '复购率(%)', '平均年龄']
# 计算各维度指标值
avg_m = [df[df['customer_value_level']==level]['M'].mean() for level in value_levels]
rep_r = [df[df['customer_value_level']==level]['is_repeat_customer'].mean()*100 for level in value_levels]
avg_a = [df[df['customer_value_level']==level]['age'].mean() for level in value_levels]

# 数据标准化（雷达图必备，统一刻度范围）
def normalize_data(data):
    return np.array(data) / np.max(data)
avg_m_norm = normalize_data(avg_m)
rep_r_norm = normalize_data(rep_r)
avg_a_norm = normalize_data(avg_a)

# 雷达图绘制配置
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# 绘制每个客户群的特征雷达图
colors_radar = ['#e74c3c', '#f39c12', '#f1c40f']
labels_radar = value_levels
for i, label in enumerate(labels_radar):
    values = np.concatenate(([avg_m_norm[i], rep_r_norm[i], avg_a_norm[i]], [avg_m_norm[i]]))
    plt.polar(angles, values, 'o-', linewidth=2, color=colors_radar[i], label=label)
    plt.fill(angles, values, alpha=0.2, color=colors_radar[i])

# 雷达图样式配置
plt.thetagrids(angles[:-1] * 180/np.pi, metrics, fontsize=14)
plt.title('🎯 客户群核心特征对比雷达图', fontsize=18, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
plt.grid(True)
plt.savefig(r"D:\桌面\数据挖掘\pythonProject1\.venv\8_客户群特征雷达图.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存：8_客户群特征雷达图.png")

# ===================== 完成提示 =====================
print("\n🎉 全部可视化完成！🎉")
print(f"📁 所有8张独立图片已保存至目录：D:\\桌面\\数据挖掘\\pythonProject1\\.venv")
print("📄 完整图片列表：")
print("  1. 1_客户价值等级分布.png")
print("  2. 2_用户聚类标签分布.png")
print("  3. 3_客户复购率分布.png")
print("  4. 4_用户年龄分布.png")
print("  5. 5_用户性别分布.png")
print("  6. 6_各价值等级平均消费.png")
print("  7. 7_聚类结果数量统计.png")
print("  8. 8_客户群特征雷达图.png")