import logging
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, APIRouter
from scipy import stats
from enum import Enum
from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("single_time_series_forecast")


class ForecastMethod(str, Enum):
    """预测方法枚举"""

    POLYNOMIAL_TREND = "polynomial_trend"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


class TimeSeriesForecastRequest(BaseModel):
    """时间序列预测请求模型

    用于对单个指标的历史数据进行未来预测。
    典型应用场景：设备参数预测、生产指标预测、趋势分析等。
    """

    data: List[float] = Field(
        ...,
        description="历史时间序列数据，至少需要5个观测值",
        min_items=5,
        example=[10.2, 10.5, 10.8, 11.1, 11.4, 11.7, 12.0, 12.3, 12.6, 12.9],
    )
    forecast_periods: Optional[int] = Field(
        default=3,
        description="预测未来的时间点数量，范围1-10",
        ge=1,
        le=10,
    )
    method: Optional[ForecastMethod] = Field(
        default=ForecastMethod.POLYNOMIAL_TREND,
        description="预测方法：polynomial_trend(多项式趋势)、exponential_smoothing(指数平滑)",
    )


class TimeSeriesForecastResponse(BaseModel):
    """时间序列预测响应模型"""

    forecast_values: List[float] = Field(..., description="预测结果数值列表")
    method: str = Field(..., description="使用的预测方法")
    data_length: int = Field(..., description="历史数据长度")
    forecast_periods: int = Field(..., description="预测期数")
    interpretation: str = Field(..., description="预测结果的解释")
    trend_direction: str = Field(..., description="预测趋势方向：上升/下降/平稳")


def calculate_polynomial_trend_forecast(
    data: np.ndarray, forecast_periods: int, confidence_level: float, degree: int = 2
) -> tuple:
    """多项式趋势预测"""
    n = len(data)
    x = np.arange(n)
    y = data

    # 使用numpy实现多项式回归
    coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coeffs)

    # 预测未来值
    future_x = np.arange(n, n + forecast_periods)
    forecast_values = poly_func(future_x)

    # 计算预测区间
    y_pred = poly_func(x)
    mse = np.mean((y - y_pred) ** 2)
    std_error = np.sqrt(mse)

    # 计算置信区间
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha / 2, n - degree - 1)
    margin_error = t_value * std_error

    lower_bounds = forecast_values - margin_error
    upper_bounds = forecast_values + margin_error

    # 计算R²作为性能指标
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return forecast_values, lower_bounds, upper_bounds, r2_score


def calculate_exponential_smoothing_forecast(
    data: np.ndarray, forecast_periods: int, confidence_level: float, alpha: float = 0.3
) -> tuple:
    """指数平滑预测"""
    n = len(data)

    # 简单指数平滑
    smoothed = np.zeros(n)
    smoothed[0] = data[0]

    for i in range(1, n):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    # 预测未来值
    forecast_values = np.full(forecast_periods, smoothed[-1])

    # 计算预测误差
    errors = data[1:] - smoothed[:-1]
    mse = np.mean(errors**2)
    std_error = np.sqrt(mse)

    # 计算置信区间
    alpha_ci = 1 - confidence_level
    z_value = stats.norm.ppf(1 - alpha_ci / 2)
    margin_error = z_value * std_error

    lower_bounds = forecast_values - margin_error
    upper_bounds = forecast_values + margin_error

    # 计算平均绝对百分比误差作为性能指标
    mape = np.mean(np.abs(errors / data[1:])) * 100

    return forecast_values, lower_bounds, upper_bounds, 100 - mape  # 转换为准确率


@router.post(
    "/api/time_series_forecast",
    response_model=TimeSeriesForecastResponse,
    operation_id="single_indicator_time_series_forecast",
    tags=["时间序列预测"],
)
async def time_series_forecast(request: TimeSeriesForecastRequest):
    """
    对单个指标的历史数据进行未来预测，适用于分钟级数据分析

    **参数说明：**
    - **data**: 历史时间序列数据，至少包含5个数据点
    - **forecast_periods**: 预测未来的时间点数量（1-10个）
    - **method**: 预测方法
        - polynomial_trend: 多项式趋势预测，适用于有非线性趋势的数据（默认）
        - exponential_smoothing: 指数平滑预测，适用于平稳波动的数据

    **返回结果：**
    - forecast_values: 预测结果数值列表
    - method: 使用的预测方法
    - data_length: 历史数据长度
    - forecast_periods: 预测期数
    - interpretation: 预测结果解释
    - trend_direction: 预测趋势方向

    **使用示例：**
    ```json
    {
        "data": [10, 11, 12, 13, 14],
        "forecast_periods": 3,
        "method": "polynomial_trend"
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data = request.data
        forecast_periods = request.forecast_periods
        method = request.method.value

        # 转换为numpy数组
        array = np.array(data)

        # 检查NaN值
        if np.any(np.isnan(array)):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据包含NaN值",
                    "message": "输入数据中包含NaN值，无法进行预测",
                    "solution": "请清理数据中的NaN值后重新提交",
                },
            )

        # 执行预测
        if method == "polynomial_trend":
            forecast_values, _, _, _ = calculate_polynomial_trend_forecast(
                array, forecast_periods, 0.95
            )
        else:  # exponential_smoothing
            forecast_values, _, _, _ = calculate_exponential_smoothing_forecast(
                array, forecast_periods, 0.95
            )

        # 判断趋势方向
        if forecast_values[-1] > array[-1]:
            trend_direction = "上升"
        elif forecast_values[-1] < array[-1]:
            trend_direction = "下降"
        else:
            trend_direction = "平稳"

        # 生成解释
        avg_change = (forecast_values[-1] - array[-1]) / forecast_periods
        if abs(avg_change) < 0.01:
            interpretation = f"预测显示指标将保持相对平稳"
        else:
            interpretation = (
                f"预测显示指标呈{trend_direction}趋势，平均每期变化{avg_change:.3f}"
            )

        # 构造响应
        return TimeSeriesForecastResponse(
            forecast_values=[float(v) for v in forecast_values],
            method=method,
            data_length=len(array),
            forecast_periods=forecast_periods,
            interpretation=interpretation,
            trend_direction=trend_direction,
        )

    except ValueError as e:
        # Pydantic验证错误
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "参数验证错误",
                "message": str(e),
                "suggestions": [
                    "检查历史数据格式是否正确",
                    "确保数据为数值类型",
                    "验证预测参数在有效范围内",
                ],
            },
        )
    except Exception as e:
        # 其他计算错误
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "计算错误",
                "message": f"时间序列预测过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的预测方法",
                ],
            },
        )
