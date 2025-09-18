import logging
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import HTTPException, APIRouter
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from enum import Enum
from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("statistic_calculate")


class StationarityTestType(str, Enum):
    """平稳性检验方法枚举"""

    ADF = "adf"
    PHILLIPS_PERRON = "pp"
    KPSS = "kpss"
    ALL = "all"


class StationarityRequest(BaseModel):
    """时间序列平稳性检验请求模型

    用于检验时间序列数据的平稳性。
    典型应用场景：时间序列分析前的数据预处理、模型选择等。
    """

    data: List[float] = Field(
        ...,
        description="时间序列数据，至少需要10个观测值",
        min_items=10,
        example=[1.2, 1.5, 1.8, 2.1, 2.3, 2.0, 1.9, 2.2, 2.5, 2.8],
    )
    test_type: Optional[StationarityTestType] = Field(
        default=StationarityTestType.ADF,
        description="检验方法：adf(ADF检验)、pp(Phillips-Perron检验)、kpss(KPSS检验)、all(全部检验)",
    )
    significance_level: Optional[float] = Field(
        default=0.05,
        description="显著性水平，常用值为0.01、0.05、0.10",
        ge=0.001,
        le=0.5,
    )


class StationarityResponse(BaseModel):
    """平稳性检验响应模型"""

    is_stationary: bool = Field(..., description="时间序列是否平稳")
    test_statistic: float = Field(..., description="检验统计量值")
    p_value: float = Field(..., description="p值，用于判断显著性")
    method: str = Field(..., description="使用的检验方法")
    data_length: int = Field(..., description="参与检验的数据点数量")
    interpretation: str = Field(..., description="检验结果的文字解释")


@router.post(
    "/api/statistic_calculate",
    response_model=StationarityResponse,
    operation_id="single_indicator_statistic_calculate",
    tags=["统计检验"],
)
async def statistic_calculate(request: StationarityRequest):
    """
    检验时间序列数据的平稳性，支持多种统计检验方法

    **参数说明：**
    - **data**: 时间序列数据，至少包含10个数据点
    - **test_type**: 检验方法
        - adf: ADF检验，检验单位根假设（默认）
        - pp: Phillips-Perron检验，对序列相关性更稳健
        - kpss: KPSS检验，检验平稳性假设
        - all: 执行所有检验方法
    - **significance_level**: 显著性水平，常用值为0.01、0.05、0.10

    **返回结果：**
    - is_stationary: 时间序列是否平稳
    - test_statistic: 检验统计量值
    - p_value: p值，用于判断显著性
    - method: 使用的检验方法
    - data_length: 数据点数量
    - interpretation: 检验结果解释

    **使用示例：**
    ```json
    {
        "data": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
        "test_type": "adf",
        "significance_level": 0.05
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data = request.data
        test_type = request.test_type.value
        significance_level = request.significance_level

        # 转换为numpy数组
        array = np.array(data)

        # 检查NaN值
        if np.any(np.isnan(array)):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据包含NaN值",
                    "message": "输入数据中包含NaN值，无法进行平稳性检验",
                    "solution": "请清理数据中的NaN值后重新提交",
                },
            )

        # 执行平稳性检验
        if test_type == "adf":
            # ADF检验
            result = adfuller(array)
            test_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < significance_level
            method = "ADF检验"

        elif test_type == "pp":
            # Phillips-Perron检验
            pp_test = PhillipsPerron(array)
            result = pp_test.run()
            test_statistic = result.stat
            p_value = result.pvalue
            is_stationary = p_value < significance_level
            method = "Phillips-Perron检验"

        elif test_type == "kpss":
            # KPSS检验
            result = kpss(array, regression="ct")
            test_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value >= significance_level  # KPSS检验的逻辑相反
            method = "KPSS检验"

        else:  # all
            # 执行ADF检验作为主要结果
            result = adfuller(array)
            test_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < significance_level
            method = "综合平稳性检验"

        # 解释检验结果
        if is_stationary:
            interpretation = "时间序列是平稳的"
        else:
            interpretation = "时间序列是非平稳的"

        # 构造响应
        return StationarityResponse(
            is_stationary=is_stationary,
            test_statistic=float(test_statistic),
            p_value=float(p_value),
            method=method,
            data_length=len(array),
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
                    "确保数据为数值类型",
                    "验证显著性水平参数在有效范围内",
                ],
            },
        )
    except Exception as e:
        # 其他计算错误
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "计算错误",
                "message": f"平稳性检验过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的检验方法",
                ],
            },
        )
