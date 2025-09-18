import logging
import numpy as np
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException, APIRouter

from config.config import *

# 配置路由和日志
router = APIRouter()
logger = logging.getLogger("outlier_detection")


class OutlierDetectionMethod(str, Enum):
    """异常值检测方法枚举"""

    IQR = "iqr"
    THREE_SIGMA = "three_sigma"
    MODIFIED_ZSCORE = "modified_zscore"


class OutlierDetectionRequest(BaseModel):
    """异常值检测请求模型
    
    用于检测数值数据中的异常值和离群点。
    典型应用场景：数据质量检查、异常值识别、数据预处理等。
    """

    data: List[float] = Field(
        ...,
        description="待检测异常值的数值数据，至少需要5个观测值",
        min_items=5,
        example=[1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1, 15.0],
    )
    method: Optional[OutlierDetectionMethod] = Field(
        default=OutlierDetectionMethod.IQR,
        description="异常值检测方法：iqr(四分位距法)、three_sigma(3西格玛法)、modified_zscore(修正Z分数法)",
    )
    sensitivity: Optional[float] = Field(
        default=1.5,
        description="检测敏感度，范围0.5-3.0，值越小检测越敏感",
        ge=0.5,
        le=3.0,
    )


class OutlierDetectionResponse(BaseModel):
    """异常值检测响应模型"""

    outliers: List[float] = Field(
        ..., description="检测到的异常值列表"
    )
    method: str = Field(..., description="实际使用的检测方法")
    data_length: int = Field(..., description="参与检测的数据点数量")
    outlier_count: int = Field(..., description="检测到的异常值数量")
    interpretation: str = Field(..., description="异常值检测结果的文字解释")


@router.post(
    "/api/outlier_detection",
    response_model=OutlierDetectionResponse,
    operation_id="single_indicator_outlier_detection",
    tags=["异常值检测"],
)
async def outlier_detection(request: OutlierDetectionRequest):
    """
    检测数值数据中的异常值，支持多种检测方法
    
    **参数说明：**
    - **data**: 数值数据，至少包含5个数据点
    - **method**: 检测方法
        - iqr: 四分位距法，基于数据的四分位数（默认）
        - three_sigma: 3西格玛法，基于均值和标准差
        - modified_zscore: 修正Z分数法，基于中位数和MAD
    - **sensitivity**: 检测敏感度，范围0.5-3.0

    **返回结果：**
    - outliers: 检测到的异常值列表
    - method: 使用的检测方法
    - data_length: 数据点数量
    - outlier_count: 异常值数量
    - interpretation: 结果解释

    **使用示例：**
    ```json
    {
        "data": [1, 2, 3, 4, 5, 100],
        "method": "iqr",
        "sensitivity": 1.5
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data = request.data
        method = request.method.value
        sensitivity = request.sensitivity

        # 转换为numpy数组
        array = np.array(data)

        # 检查NaN值
        if np.any(np.isnan(array)):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据包含NaN值",
                    "message": "输入数据中包含NaN值，无法进行异常值检测",
                    "solution": "请清理数据中的NaN值后重新提交",
                },
            )

        # 执行异常值检测
        outliers = []

        if method == "iqr":
            # IQR方法
            q1, q3 = np.percentile(array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - sensitivity * iqr
            upper_bound = q3 + sensitivity * iqr
            
            outlier_mask = (array < lower_bound) | (array > upper_bound)
            outliers = array[outlier_mask].tolist()

        elif method == "three_sigma":
            # 3西格玛方法
            mean_val = np.mean(array)
            std_val = np.std(array)
            threshold = sensitivity
            
            z_scores = np.abs((array - mean_val) / std_val)
            outlier_mask = z_scores > threshold
            outliers = array[outlier_mask].tolist()

        elif method == "modified_zscore":
            # 修正Z分数方法
            median_val = np.median(array)
            mad = np.median(np.abs(array - median_val))
            
            if mad == 0:
                mad = np.std(array)
            
            modified_z_scores = 0.6745 * (array - median_val) / mad
            threshold = sensitivity + 2.0
            
            outlier_mask = np.abs(modified_z_scores) > threshold
            outliers = array[outlier_mask].tolist()

        # 解释异常值检测结果
        outlier_count = len(outliers)
        data_length = len(array)
        outlier_percentage = (outlier_count / data_length) * 100
        
        if outlier_count == 0:
            interpretation = "未检测到异常值"
        elif outlier_percentage < 5:
            interpretation = "检测到少量异常值"
        elif outlier_percentage < 15:
            interpretation = "检测到适量异常值"
        else:
            interpretation = "检测到大量异常值"

        # 构造响应
        return OutlierDetectionResponse(
            outliers=outliers,
            method=method,
            data_length=data_length,
            outlier_count=outlier_count,
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
                    "验证敏感度参数在有效范围内",
                ],
            },
        )
    except Exception as e:
        # 其他计算错误
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "计算错误",
                "message": f"异常值检测过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的检测方法",
                ],
            },
        )
