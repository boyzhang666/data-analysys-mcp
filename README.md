# 数据分析MCP工具集

专业的工业数据分析工具集，基于FastAPI-MCP架构，提供多种统计分析方法，适用于时间序列分析、数据质量检查和统计建模等场景。

## 🚀 功能特性

### 核心分析工具
- **相关性分析**: 支持Pearson、Spearman、Kendall相关系数计算
- **平稳性检验**: 提供ADF、PP、KPSS时间序列平稳性检验
- **分布分析**: 数据分布特征和趋势模式分析
- **异常值检测**: 多种异常值检测方法和综合评估
- **因果关系分析**: 格兰杰因果检验、互相关分析等（支持两变量和多变量分析）
- **时序相似度分析**: DTW动态时间规整、滑动窗口分析等
- **时间序列预测**: 多项式趋势预测、指数平滑预测（专为分钟级数据优化）

### 技术特点
- 基于FastAPI-MCP架构，支持AI工具调用
- RESTful API接口，易于集成
- 完整的数据验证和错误处理
- 详细的分析结果和建议
- 支持多种数据格式和参数配置

## 📁 项目结构

```
data_analysis_mcp/
├── config/                           # 配置文件目录
│   ├── __init__.py
│   ├── config.json                  # 配置文件
│   └── config.py                    # 配置加载模块
├── routers/                         # 路由模块目录
│   ├── two_correlation_calculate.py        # 两变量相关性分析
│   ├── single_statistic_calculate.py       # 单变量平稳性检验
│   ├── single_distribution_analysis.py     # 单变量分布分析
│   ├── single_outlier_detection.py         # 单变量异常值检测
│   ├── two_causal_analysis.py              # 两变量因果关系分析
│   ├── multi_causal_analysis.py            # 多变量因果关系分析
│   ├── two_similarity_analysis.py          # 两变量时序相似度分析
│   ├── single_time_series_forecast.py      # 单变量时间序列预测
│   └── utils/                              # 工具函数目录
├── log/                             # 日志文件目录
├── doc/                             # 文档目录
├── main.py                          # 应用启动文件
├── requirements.txt                 # 依赖包列表
└── README.md                        # 项目说明文档
```

## 🛠️ 安装和使用

### 环境要求
- Python 3.8+
- FastAPI
- NumPy, SciPy, Pandas
- Statsmodels

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务
```bash
python main.py
```

服务将在 `http://localhost:6003` 启动。

### API文档
启动服务后，访问以下地址查看API文档：
- Swagger UI: `http://localhost:6003/docs`
- ReDoc: `http://localhost:6003/redoc`
- MCP工具列表: `http://localhost:6003/mcp/tools`

## 📊 API接口说明

### 1. 两变量相关性分析 (`/api/correlation`)
计算两组数据间的相关系数，支持Pearson、Spearman、Kendall相关性度量方法。

**请求示例：**
```json
{
    "data1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "data2": [2.0, 4.0, 6.0, 8.0, 10.0],
    "method": "pearson"
}
```

### 2. 单变量平稳性检验 (`/api/statistic_calculate`)
执行时间序列平稳性检验，支持ADF、PP、KPSS检验方法。

**请求示例：**
```json
{
    "data": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1],
    "test_type": "adf",
    "significance_level": 0.05
}
```

### 3. 单变量分布分析 (`/api/distribution_analysis`)
分析单个变量的分布特征和趋势模式。

**请求示例：**
```json
{
    "data": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1]
}
```

### 4. 单变量异常值检测 (`/api/outlier_detection`)
使用多种方法检测单个变量数据中的异常值。

**请求示例：**
```json
{
    "data": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1, 15.0],
    "method": "iqr"
}
```

### 5. 两变量因果关系分析 (`/api/causal_analysis`)
分析两个变量间的因果关系，支持格兰杰因果检验等方法。

**请求示例：**
```json
{
    "data1": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1],
    "data2": [2.1, 3.2, 2.5, 5.1, 2.8, 4.1, 2.6, 3.5, 4.2, 6.8],
    "method": "granger",
    "max_lag": 3
}
```

### 6. 多变量因果关系分析 (`/api/multi_causal_analysis`)
分析多个测点之间的因果关系网络，识别影响关系。

**请求示例：**
```json
{
    "data": {
        "temperature": [20.1, 20.5, 21.0, 21.2, 21.8, 22.0, 21.5, 21.0, 20.8, 20.3],
        "pressure": [1.01, 1.02, 1.05, 1.08, 1.12, 1.15, 1.10, 1.06, 1.03, 1.01],
        "flow_rate": [10.2, 10.8, 11.5, 12.0, 12.8, 13.2, 12.5, 11.8, 11.0, 10.5]
    },
    "method": "granger"
}
```

### 7. 两变量时序相似度分析 (`/api/similarity_analysis`)
分析两个时间序列的相似度和最佳匹配区间。

**请求示例：**
```json
{
    "data1": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1],
    "data2": [1.5, 2.1, 2.0, 4.2, 2.3, 3.0, 2.1, 2.9, 3.3, 5.8],
    "window_size": 5,
    "normalize": true
}
```

### 8. 单变量时间序列预测 (`/api/time_series_forecast`)
对单个指标的历史数据进行未来预测，专为分钟级数据优化。

**请求示例：**
```json
{
    "data": [10, 11, 12, 13, 14],
    "forecast_periods": 3,
    "method": "polynomial_trend"
}
```

## 🔧 配置说明

### 配置文件 (`config/config.json`)
```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 6003
    },
    "logging": {
        "level": "INFO",
        "max_bytes": 3145728,
        "backup_count": 1
    }
}
```

## 📝 开发说明

### 添加新的分析工具
1. 在 `routers/` 目录下创建新的路由文件
2. 定义请求和响应模型
3. 实现分析逻辑
4. 在 `main.py` 中注册路由
5. 更新MCP配置中的 `include_operations`

### 代码规范
- 使用Pydantic进行数据验证
- 添加详细的API文档字符串
- 实现完整的错误处理
- 添加日志记录
- 编写单元测试

## 📈 应用场景

### 工业应用
- 设备状态监控和异常检测
- 过程参数相关性分析
- 生产数据质量评估
- 预测模型特征工程

### 数据科学
- 探索性数据分析(EDA)
- 时间序列分析
- 统计建模预处理
- 数据质量评估

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 项目讨论区

---

**版本**: v1.0.0  
**最后更新**: 2024年1月