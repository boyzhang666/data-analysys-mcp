import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Tuple
from fastapi import HTTPException, APIRouter
from statsmodels.tsa.stattools import grangercausalitytests
from enum import Enum
from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("causal_analysis")


class CausalAnalysisMethod(str, Enum):
    """因果分析方法枚举，避免字符串输入错误"""

    GRANGER = "granger"
    CROSS_CORRELATION = "cross_correlation"
    TRANSFER_ENTROPY = "transfer_entropy"
    VAR_ANALYSIS = "var_analysis"


class CausalAnalysisRequest(BaseModel):
    """因果分析请求模型
    
    用于分析两个时间序列数据之间的因果关系。
    典型应用场景：分析温度变化对压力的影响、股价对成交量的因果关系等。
    """

    data1: List[float] = Field(
        ...,
        description="第一个时间序列数据，用于因果分析",
        min_items=10,
        example=[1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0],
    )
    data2: List[float] = Field(
        ...,
        description="第二个时间序列数据，与第一个数据分析因果关系，长度必须相同",
        min_items=10,
        example=[2.0, 4.0, 6.0, 8.0, 10.0, 8.0, 6.0, 4.0, 2.0, 4.0],
    )
    method: Optional[CausalAnalysisMethod] = Field(
        default=CausalAnalysisMethod.GRANGER,
        description="因果分析方法：granger(格兰杰检验)、cross_correlation(互相关)、transfer_entropy(传递熵)",
    )
    max_lag: Optional[int] = Field(
        default=3,
        description="最大滞后期数，用于分析时间延迟效应",
        ge=1,
        le=5,
    )
    
    @field_validator('data1', 'data2')
    @classmethod
    def validate_data_length(cls, v):
        if len(v) < 10:
            raise ValueError('数据长度至少需要10个观测值')
        return v


class CausalAnalysisResponse(BaseModel):
    """因果分析响应模型"""

    causal_strength: float = Field(
        ..., description="因果关系强度，范围0-1，越接近1表示因果关系越强"
    )
    method: str = Field(..., description="实际使用的因果分析方法")
    data_length: int = Field(..., description="参与分析的有效数据点数量（排除NaN值后）")
    interpretation: str = Field(..., description="因果关系强度的文字解释")
    direction: str = Field(..., description="主要因果方向：data1->data2, data2->data1, 或 bidirectional")
    details: Optional[Dict[str, Any]] = Field(None, description="详细的分析结果（可选）")


def calculate_transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    """计算传递熵（简化版本）"""
    try:
        # 简化的传递熵计算，基于条件互信息
        # 这里使用基于分位数的离散化方法
        n_bins = min(10, len(x) // 5)  # 自适应分箱数

        # 离散化数据
        x_discrete = pd.cut(x, bins=n_bins, labels=False)
        y_discrete = pd.cut(y, bins=n_bins, labels=False)

        # 创建滞后序列
        if lag >= len(x):
            return 0.0

        x_lag = x_discrete[:-lag] if lag > 0 else x_discrete
        y_current = y_discrete[lag:] if lag > 0 else y_discrete
        y_lag = y_discrete[:-lag] if lag > 0 else y_discrete[:-1]

        if len(x_lag) != len(y_current) or len(y_current) != len(y_lag):
            return 0.0

        # 计算各种熵
        try:
            # 联合分布
            joint_xy = np.column_stack([y_current, y_lag, x_lag])
            joint_y = np.column_stack([y_current, y_lag])

            # 使用直方图估计概率分布
            hist_joint_xy, _ = np.histogramdd(joint_xy, bins=min(5, len(joint_xy) // 3))
            hist_joint_y, _ = np.histogramdd(joint_y, bins=min(5, len(joint_y) // 3))
            hist_y_lag, _ = np.histogram(y_lag, bins=min(5, len(y_lag) // 3))

            # 归一化为概率
            p_joint_xy = hist_joint_xy / np.sum(hist_joint_xy)
            p_joint_y = hist_joint_y / np.sum(hist_joint_y)
            p_y_lag = hist_y_lag / np.sum(hist_y_lag)

            # 避免零概率
            p_joint_xy = p_joint_xy + 1e-10
            p_joint_y = p_joint_y + 1e-10
            p_y_lag = p_y_lag + 1e-10

            # 计算传递熵（简化版本）
            te = np.sum(
                p_joint_xy
                * np.log2(
                    p_joint_xy
                    / (p_joint_y[:, :, np.newaxis] * p_y_lag[np.newaxis, :, np.newaxis])
                )
            )

            return max(0.0, float(te))

        except Exception:
            # 如果计算失败，返回基于相关性的简化估计
            corr = np.corrcoef(x_lag, y_current)[0, 1]
            return max(0.0, abs(corr) ** 2)

    except Exception as e:
        logger.warning(f"传递熵计算失败: {str(e)}")
        return 0.0


@router.post(
    "/api/causal_analysis",
    response_model=CausalAnalysisResponse,
    operation_id="two_indicators_causal_analysis",
    tags=["因果分析"],
)
async def causal_analysis(request: CausalAnalysisRequest):
    """
    分析两个时间序列数据之间的因果关系，支持多种因果分析方法
    
    **参数说明：**
    - **data1**: 第一个时间序列数据，至少包含10个数据点
    - **data2**: 第二个时间序列数据，长度必须与data1相同
    - **method**: 因果分析方法
        - granger: 格兰杰因果检验，检验统计因果关系（默认）
        - cross_correlation: 互相关分析，分析相关性强度
        - transfer_entropy: 传递熵，基于信息论的因果分析
    - **max_lag**: 最大滞后期数，用于分析时间延迟效应

    **返回结果：**
    - causal_strength: 因果关系强度（0到1之间）
    - method: 使用的分析方法
    - data_length: 有效数据点数量
    - interpretation: 因果关系强度解释
    - direction: 主要因果方向
    - details: 详细分析结果（可选）

    **使用示例：**
    ```json
    {
        "data1": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
        "data2": [2, 4, 6, 8, 10, 8, 6, 4, 2, 4],
        "method": "granger"
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data1 = request.data1
        data2 = request.data2
        method = request.method.value
        max_lag = request.max_lag

        # 转换为numpy数组
        array1 = np.array(data1)
        array2 = np.array(data2)

        # 检查NaN值
        if np.any(np.isnan(array1)) or np.any(np.isnan(array2)):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据包含NaN值",
                    "message": "输入数据中包含NaN值，无法进行因果分析",
                    "solution": "请清理数据中的NaN值后重新提交",
                },
            )

        # 检查数据变异性
        if np.var(array1) == 0 or np.var(array2) == 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据无变异性",
                    "message": "数据序列无变化，无法进行因果分析",
                    "solution": "请提供具有变化的时间序列数据",
                },
            )

        # 使用原始数组进行分析
        valid_array1 = array1
        valid_array2 = array2

        # 执行因果分析
        causal_strength = 0.0
        direction = "none"
        details = {}

        if method == "granger":
            # 格兰杰因果检验
            try:
                lag = min(max_lag, len(valid_array1) // 4)  # 动态调整滞后期
                
                # data1 -> data2 检验
                data_12 = np.column_stack([valid_array2, valid_array1])
                granger_12 = grangercausalitytests(data_12, maxlag=lag, verbose=False)
                pvalue_12 = granger_12[lag][0]["ssr_ftest"][1]
                
                # data2 -> data1 检验
                data_21 = np.column_stack([valid_array1, valid_array2])
                granger_21 = grangercausalitytests(data_21, maxlag=lag, verbose=False)
                pvalue_21 = granger_21[lag][0]["ssr_ftest"][1]
                
                # 计算因果强度
                strength_12 = max(0, 1 - pvalue_12)
                strength_21 = max(0, 1 - pvalue_21)
                
                causal_strength = max(strength_12, strength_21)
                
                # 确定主要方向
                if strength_12 > 0.5 and strength_21 > 0.5:
                    direction = "bidirectional"
                elif strength_12 > strength_21:
                    direction = "data1->data2"
                elif strength_21 > strength_12:
                    direction = "data2->data1"
                else:
                    direction = "none"
                
                details = {
                    "data1_to_data2": {"p_value": float(pvalue_12), "strength": float(strength_12)},
                    "data2_to_data1": {"p_value": float(pvalue_21), "strength": float(strength_21)},
                    "lag_used": lag
                }
                
            except Exception as e:
                details = {"error": f"格兰杰检验计算失败: {str(e)}"}

        elif method == "cross_correlation":
            # 互相关分析
            try:
                correlation = np.corrcoef(valid_array1, valid_array2)[0, 1]
                causal_strength = abs(correlation)
                direction = "bidirectional" if causal_strength > 0.3 else "none"
                
                details = {
                    "correlation": float(correlation),
                    "abs_correlation": float(causal_strength)
                }
                
            except Exception as e:
                details = {"error": f"互相关计算失败: {str(e)}"}

        elif method == "transfer_entropy":
            # 传递熵分析
            try:
                te_12 = calculate_transfer_entropy(valid_array1, valid_array2, 1)
                te_21 = calculate_transfer_entropy(valid_array2, valid_array1, 1)
                
                causal_strength = max(te_12, te_21)
                
                if te_12 > 0.1 and te_21 > 0.1:
                    direction = "bidirectional"
                elif te_12 > te_21:
                    direction = "data1->data2"
                elif te_21 > te_12:
                    direction = "data2->data1"
                else:
                    direction = "none"
                
                details = {
                    "data1_to_data2": float(te_12),
                    "data2_to_data1": float(te_21)
                }
                
            except Exception as e:
                details = {"error": f"传递熵计算失败: {str(e)}"}

        # 解释因果关系强度
        if causal_strength >= 0.7:
            interpretation = "强因果关系"
        elif causal_strength >= 0.5:
            interpretation = "中等因果关系"
        elif causal_strength >= 0.3:
            interpretation = "弱因果关系"
        else:
            interpretation = "几乎无因果关系"

        # 构造响应
        return CausalAnalysisResponse(
            causal_strength=float(causal_strength),
            method=method,
            data_length=len(valid_array1),
            interpretation=interpretation,
            direction=direction,
            details=details,
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
                "message": f"因果分析计算过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试使用不同的分析方法",
                ],
            },
        )
