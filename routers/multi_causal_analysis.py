import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, APIRouter
from statsmodels.tsa.stattools import grangercausalitytests
from pydantic import BaseModel, Field, field_validator

from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("multi_causal_analysis")


class MultiCausalAnalysisMethod(str, Enum):
    """多变量因果分析方法枚举"""

    GRANGER = "granger"
    CORRELATION_MATRIX = "correlation_matrix"
    VAR_ANALYSIS = "var_analysis"


class MultiCausalAnalysisRequest(BaseModel):
    """多测点因果分析请求模型

    用于分析多个时间序列数据之间的因果关系。
    典型应用场景：分析多个传感器之间的影响关系、识别关键影响因子等。
    """

    data: Dict[str, List[float]] = Field(
        ...,
        description="多个测点的时间序列数据，键为测点名称，值为数据序列",
        example={
            "temperature": [20.1, 20.5, 21.0, 21.2, 21.8, 22.0, 21.5, 21.0, 20.8, 20.3],
            "pressure": [1.01, 1.02, 1.05, 1.08, 1.12, 1.15, 1.10, 1.06, 1.03, 1.01],
            "flow_rate": [10.2, 10.8, 11.5, 12.0, 12.8, 13.2, 12.5, 11.8, 11.0, 10.5],
        },
    )
    method: Optional[MultiCausalAnalysisMethod] = Field(
        default=MultiCausalAnalysisMethod.GRANGER,
        description="分析方法：granger(格兰杰因果检验)、correlation_matrix(相关性矩阵)、var_analysis(VAR分析)",
    )
    max_lag: Optional[int] = Field(
        default=3,
        description="最大滞后期数，用于分析时间延迟效应",
        ge=1,
        le=5,
    )
    significance_level: Optional[float] = Field(
        default=0.05,
        description="显著性水平，用于判断因果关系的显著性",
        ge=0.001,
        le=0.5,
    )

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if len(v) < 2:
            raise ValueError("至少需要2个测点的数据")
        if len(v) > 10:
            raise ValueError("最多支持10个测点的分析")

        # 检查数据长度一致性
        lengths = [len(series) for series in v.values()]
        if len(set(lengths)) > 1:
            raise ValueError("所有测点的数据长度必须相同")

        # 检查最小数据长度
        min_length = min(lengths)
        if min_length < 10:
            raise ValueError("每个测点至少需要10个观测值")

        return v


class MultiCausalAnalysisResponse(BaseModel):
    """多测点因果分析响应模型"""

    causal_matrix: Dict[str, Dict[str, float]] = Field(
        ..., description="因果关系矩阵，显示各测点间的因果强度"
    )
    significant_relationships: List[Dict[str, Any]] = Field(
        ..., description="显著的因果关系列表"
    )
    method: str = Field(..., description="使用的分析方法")
    data_info: Dict[str, Any] = Field(..., description="数据基本信息")
    interpretation: str = Field(..., description="分析结果的总体解释")
    recommendations: List[str] = Field(..., description="基于分析结果的建议")


def calculate_granger_causality_matrix(
    data_dict: Dict[str, np.ndarray], max_lag: int, significance_level: float
) -> Dict[str, Dict[str, float]]:
    """计算格兰杰因果关系矩阵"""
    variables = list(data_dict.keys())
    causal_matrix = {}

    for i, var1 in enumerate(variables):
        causal_matrix[var1] = {}
        for j, var2 in enumerate(variables):
            if i == j:
                causal_matrix[var1][var2] = 0.0  # 自己对自己的因果关系设为0
            else:
                try:
                    # 构造数据：var2作为因变量，var1作为自变量
                    test_data = np.column_stack([data_dict[var2], data_dict[var1]])

                    # 执行格兰杰因果检验
                    result = grangercausalitytests(
                        test_data, maxlag=max_lag, verbose=False
                    )

                    # 获取最优滞后期的p值
                    p_values = [
                        result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                    ]
                    min_p_value = min(p_values)

                    # 计算因果强度（1 - p值）
                    causal_strength = (
                        max(0, 1 - min_p_value)
                        if min_p_value < significance_level
                        else 0.0
                    )
                    causal_matrix[var1][var2] = causal_strength
                except Exception as e:
                    logger.warning(
                        f"计算{var1}->{var2}的格兰杰因果关系时出错: {str(e)}"
                    )
                    causal_matrix[var1][var2] = 0.0

    return causal_matrix


def calculate_correlation_matrix(
    data_dict: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """计算相关性矩阵"""
    variables = list(data_dict.keys())
    correlation_matrix = {}

    for var1 in variables:
        correlation_matrix[var1] = {}
        for var2 in variables:
            if var1 == var2:
                correlation_matrix[var1][var2] = 1.0
            else:
                try:
                    corr = np.corrcoef(data_dict[var1], data_dict[var2])[0, 1]
                    correlation_matrix[var1][var2] = (
                        abs(corr) if not np.isnan(corr) else 0.0
                    )
                except Exception:
                    correlation_matrix[var1][var2] = 0.0

    return correlation_matrix


def identify_significant_relationships(
    causal_matrix: Dict[str, Dict[str, float]], threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """识别显著的因果关系"""
    significant_relationships = []

    for cause, effects in causal_matrix.items():
        for effect, strength in effects.items():
            if strength > threshold:
                relationship = {
                    "cause": cause,
                    "effect": effect,
                    "strength": strength,
                    "significance": (
                        "强" if strength > 0.7 else "中等" if strength > 0.5 else "弱"
                    ),
                }
                significant_relationships.append(relationship)

    # 按强度排序
    significant_relationships.sort(key=lambda x: x["strength"], reverse=True)

    return significant_relationships


@router.post(
    "/api/multi_causal_analysis",
    response_model=MultiCausalAnalysisResponse,
    operation_id="multi_indicators_causal_analysis",
    tags=["多变量因果分析"],
)
async def multi_causal_analysis(request: MultiCausalAnalysisRequest):
    """
    分析多个测点之间的因果关系，识别影响关系网络

    **参数说明：**
    - **data**: 多个测点的时间序列数据字典
    - **method**: 分析方法
        - granger: 格兰杰因果检验，分析时间序列因果关系（默认）
        - correlation_matrix: 相关性矩阵分析
        - var_analysis: VAR模型分析
    - **max_lag**: 最大滞后期数，用于分析时间延迟效应
    - **significance_level**: 显著性水平，用于判断关系的显著性

    **返回结果：**
    - causal_matrix: 因果关系矩阵
    - significant_relationships: 显著的因果关系列表
    - method: 使用的分析方法
    - data_info: 数据基本信息
    - interpretation: 分析结果解释
    - recommendations: 实用建议

    **使用示例：**
    ```json
    {
        "data": {
            "temperature": [20, 21, 22, 23, 24, 23, 22, 21, 20, 21],
            "pressure": [1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1]
        },
        "method": "granger"
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data_dict = request.data
        method = request.method.value
        max_lag = request.max_lag
        significance_level = request.significance_level

        # 转换为numpy数组
        np_data = {}
        for var_name, var_data in data_dict.items():
            array = np.array(var_data)

            # 检查NaN值
            if np.any(np.isnan(array)):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_type": "数据包含NaN值",
                        "message": f"测点'{var_name}'的数据中包含NaN值，无法进行因果分析",
                        "solution": "请清理数据中的NaN值后重新提交",
                    },
                )

            np_data[var_name] = array

        # 执行因果分析
        if method == "granger":
            causal_matrix = calculate_granger_causality_matrix(
                np_data, max_lag, significance_level
            )
        elif method == "correlation_matrix":
            causal_matrix = calculate_correlation_matrix(np_data)
        else:  # var_analysis
            # 简化的VAR分析，使用格兰杰因果检验作为基础
            causal_matrix = calculate_granger_causality_matrix(
                np_data, max_lag, significance_level
            )

        # 识别显著关系
        significant_relationships = identify_significant_relationships(causal_matrix)

        # 数据信息
        variables = list(data_dict.keys())
        data_length = len(list(data_dict.values())[0])
        data_info = {
            "variables": variables,
            "variable_count": len(variables),
            "data_length": data_length,
            "total_relationships": len(variables) * (len(variables) - 1),
        }

        # 生成解释
        if significant_relationships:
            strong_relationships = [
                r for r in significant_relationships if r["strength"] > 0.7
            ]
            if strong_relationships:
                interpretation = f"发现{len(strong_relationships)}个强因果关系和{len(significant_relationships) - len(strong_relationships)}个中等/弱因果关系"
            else:
                interpretation = f"发现{len(significant_relationships)}个中等/弱因果关系，无强因果关系"
        else:
            interpretation = "未发现显著的因果关系"

        # 生成建议
        recommendations = []
        if significant_relationships:
            # 找出最重要的影响因子
            cause_counts = {}
            for rel in significant_relationships:
                cause = rel["cause"]
                cause_counts[cause] = cause_counts.get(cause, 0) + 1

            if cause_counts:
                top_cause = max(cause_counts.items(), key=lambda x: x[1])[0]
                recommendations.append(
                    f"'{top_cause}'是最重要的影响因子，影响{cause_counts[top_cause]}个其他测点"
                )

            recommendations.append("建议重点监控具有强因果关系的测点组合")
            recommendations.append("可以利用因果关系进行预测性维护和异常检测")
        else:
            recommendations.append("各测点之间相对独立，可以分别进行监控和分析")
            recommendations.append("建议检查数据质量或尝试不同的分析方法")

        # 构造响应
        return MultiCausalAnalysisResponse(
            causal_matrix=causal_matrix,
            significant_relationships=significant_relationships,
            method=f"{method}分析",
            data_info=data_info,
            interpretation=interpretation,
            recommendations=recommendations,
        )

    except ValueError as e:
        # Pydantic验证错误
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "参数验证错误",
                "message": str(e),
                "suggestions": [
                    "检查数据格式是否正确",
                    "确保所有测点数据长度相同",
                    "验证测点数量在2-10个范围内",
                ],
            },
        )
    except Exception as e:
        # 其他计算错误
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "计算错误",
                "message": f"多测点因果分析过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的分析方法",
                ],
            },
        )
