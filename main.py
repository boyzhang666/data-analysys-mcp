import uvicorn
import logging
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from routers import (
    single_statistic_calculate,
    single_distribution_analysis,
    single_outlier_detection,
    two_correlation_calculate,
    two_causal_analysis,
    two_similarity_analysis,
    multi_causal_analysis,
    single_time_series_forecast,
)
from logging.handlers import RotatingFileHandler

# 创建日志记录器字典
sub_route_files = {}
sub_route_files_list = [
    "single_statistic_calculate",
    "single_distribution_analysis",
    "single_outlier_detection",
    "two_correlation_calculate",
    "two_causal_analysis",
    "two_similarity_analysis",
    "multi_causal_analysis",
    "single_time_series_forecast",
]

# 配置日志记录器
for sub_route_file in sub_route_files_list:
    sub_route_logger = logging.getLogger(sub_route_file)
    sub_route_logger.setLevel(logging.INFO)
    sub_route_logger.propagate = False

    if not sub_route_logger.handlers:
        sub_route_file_handle = RotatingFileHandler(
            f"log/{sub_route_file}.log",
            encoding="UTF-8",
            maxBytes=1024 * 1024 * 3,
            backupCount=1,
        )
        sub_route_file_handle.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [line:%(lineno)d] -- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sub_route_file_handle.setFormatter(fmt)
        sub_route_logger.addHandler(sub_route_file_handle)

    sub_route_files[sub_route_file] = sub_route_logger

# 创建 FastAPI 应用
app = FastAPI(title="Industrial Plant Data API")


# 创建 MCP 包装器
mcp = FastApiMCP(
    app,
    name="data_analysis_mcp",
    description="专业的数据分析工具集，提供数据趋势分析、质量分析、平稳性检验、时间序列分析、相关性分析、因果关系分析等功能。",
    describe_full_response_schema=True,
    auth_config=None,
    include_operations=[
        "two_indicators_correlation_calculate",
        "single_indicator_statistic_calculate",
        "single_indicator_distribution_analysis",
        "single_indicator_outlier_detection",
        "two_indicators_causal_analysis",
        "two_indicators_similarity_analysis",
        "multi_indicators_causal_analysis",
        "single_indicator_time_series_forecast",
    ],
)

# 包含路由
app.include_router(single_distribution_analysis.router, tags=["趋势分析"])
app.include_router(single_outlier_detection.router, tags=["异常值检测"])
app.include_router(single_statistic_calculate.router, tags=["平稳性检验"])
app.include_router(two_correlation_calculate.router, tags=["相关性分析"])
app.include_router(two_causal_analysis.router, tags=["因果关系分析"])
app.include_router(two_similarity_analysis.router, tags=["相似度分析"])
app.include_router(multi_causal_analysis.router, tags=["多因果分析"])
app.include_router(single_time_series_forecast.router, tags=["时间序列预测"])


# 挂载 MCP 服务
mcp.mount_http()
# 刷新MCP服务器以包含新端点
mcp.setup_server()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6003)
