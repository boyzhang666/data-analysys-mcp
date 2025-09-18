import logging
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from fastapi import HTTPException, APIRouter
from enum import Enum
from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("correlation_calculate")


class CorrelationMethod(str, Enum):
    """相关性计算方法枚举，避免字符串输入错误"""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationRequest(BaseModel):
    """相关性计算请求模型

    用于计算两组数值数据之间的相关性强度。
    典型应用场景：分析温度与压力的关联性、股价与成交量的相关性等。
    """

    data1: List[float] = Field(
        ...,
        description="第一组数值数据，用于计算相关性",
        min_items=2,
        example=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    data2: List[float] = Field(
        ...,
        description="第二组数值数据，与第一组数据计算相关性，长度必须相同",
        min_items=2,
        example=[2.0, 4.0, 6.0, 8.0, 10.0],
    )
    method: Optional[CorrelationMethod] = Field(
        default=CorrelationMethod.PEARSON,
        description="相关性计算方法：pearson(线性相关)、spearman(单调相关)、kendall(秩相关)",
    )


class CorrelationResponse(BaseModel):
    """相关性计算响应模型"""

    correlation: float = Field(
        ..., description="相关系数值，范围通常在-1到1之间，越接近±1表示相关性越强"
    )
    method: str = Field(..., description="实际使用的相关性计算方法")
    data_length: int = Field(..., description="参与计算的有效数据点数量（排除NaN值后）")
    interpretation: str = Field(..., description="相关性强度的文字解释")


@router.post(
    "/api/correlation",
    response_model=CorrelationResponse,
    operation_id="two_indicators_correlation_calculate",
    tags=["相关性分析"],
)
async def correlation_calculate(request: CorrelationRequest):
    """
    计算两组数据之间的相关系数，支持多种相关性分析方法
    
    **参数说明：**
    - **data1**: 第一组数值数据，至少包含2个数据点
    - **data2**: 第二组数值数据，长度必须与data1相同
    - **method**: 相关性计算方法
        - pearson: 皮尔逊相关系数，衡量线性相关性（默认）
        - spearman: 斯皮尔曼秩相关系数，衡量单调相关性
        - kendall: 肯德尔τ相关系数，基于秩次的相关性

    **返回结果：**
    - correlation: 相关系数（-1到1之间）
    - method: 使用的计算方法
    - data_length: 有效数据点数量
    - interpretation: 相关性强度解释

    **使用示例：**
    ```json
    {
        "data1": [1, 2, 3, 4, 5],
        "data2": [2, 4, 6, 8, 10],
        "method": "pearson"
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data1 = request.data1
        data2 = request.data2
        method = request.method.value  # 获取枚举值

        # 转换为numpy数组
        array1 = np.array(data1)
        array2 = np.array(data2)

        # 处理NaN值
        valid_mask = ~(np.isnan(array1) | np.isnan(array2))
        valid_array1 = array1[valid_mask]
        valid_array2 = array2[valid_mask]

        if len(valid_array1) < 2:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "有效数据不足",
                    "message": f"清理NaN值后只剩{len(valid_array1)}个有效数据点，至少需要2个",
                    "original_length": len(array1),
                    "valid_length": len(valid_array1),
                    "solution": "请提供更多有效数据或清理数据中的NaN值",
                },
            )

        # 计算相关性
        if method == "pearson":
            # 皮尔逊相关系数
            correlation = np.corrcoef(valid_array1, valid_array2)[0, 1]
        elif method == "spearman":
            # 斯皮尔曼秩相关系数
            from scipy import stats

            correlation, _ = stats.spearmanr(valid_array1, valid_array2)
        else:  # kendall
            # 肯德尔秩相关系数
            from scipy import stats

            correlation, _ = stats.kendalltau(valid_array1, valid_array2)

        # 解释相关性强度
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            interpretation = "强相关"
        elif abs_corr >= 0.6:
            interpretation = "中等相关"
        elif abs_corr >= 0.3:
            interpretation = "弱相关"
        else:
            interpretation = "几乎无相关"

        if correlation < 0:
            interpretation = "负" + interpretation
        elif correlation > 0:
            interpretation = "正" + interpretation
        else:
            interpretation = "无相关"

        # 构造响应
        return CorrelationResponse(
            correlation=float(correlation),
            method=method,
            data_length=len(valid_array1),
            interpretation=interpretation,
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
                    "确保两组数据长度相同",
                    "验证数据中没有NaN或无穷大值",
                ],
            },
        )
    except Exception as e:
        # 其他计算错误
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "计算错误",
                "message": f"相关性计算过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的相关性方法",
                ],
            },
        )
