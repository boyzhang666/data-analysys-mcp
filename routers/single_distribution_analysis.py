import logging
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException, APIRouter

from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("distribution_analysis")


class DistributionRequest(BaseModel):
    """数据分布趋势分析请求模型"""

    data: List[float] = Field(
        ...,
        description="待分析的数值数据，至少需要5个观测值",
        min_items=5,
        example=[1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1],
    )


class DistributionResponse(BaseModel):
    """数据分布趋势分析响应模型"""

    quartiles: dict = Field(
        ..., description="四分位数信息，包含Q1、Q2(中位数)、Q3和IQR"
    )
    statistics: dict = Field(
        ..., description="基本统计信息，包含均值、标准差、数据量等"
    )
    distribution_summary: str = Field(..., description="数据分布特征的文字总结")
    trend_analysis: str = Field(..., description="数据趋势分析结果")
    shape_characteristics: dict = Field(
        ..., description="分布形状特征，包含偏度、峰度等"
    )


@router.post(
    "/api/distribution_analysis",
    response_model=DistributionResponse,
    operation_id="single_indicator_distribution_analysis",
)
async def distribution_analysis(request: DistributionRequest):
    """
    执行数据分布趋势分析，专注于数据的分布特征和趋势模式

    **适用场景：**
    - 探索性数据分析(EDA)的分布特征分析
    - 数据趋势识别和模式发现
    - 统计报告中的描述性统计分析
    - 数据分布形状特征评估

    **分析方法：**
    - 基于四分位数(Q1, Q2, Q3)的分布分析
    - 计算偏度和峰度等形状特征
    - 提供完整的描述性统计信息
    - 分析数据的集中趋势和离散程度

    **参数说明：**
    - **data**: 待分析的数值数据，建议至少5个观测值

    **返回信息：**
    - 四分位数统计
    - 基本统计信息
    - 数据分布特征总结
    - 趋势分析结果
    - 分布形状特征

    **使用示例：**
    ```json
    {
        "data": [1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1]
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data = np.array(request.data)

        if len(data) < 5:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据不足",
                    "message": f"当前数据只有{len(data)}个观测值，分布分析至少需要5个",
                    "current_length": len(data),
                    "minimum_required": 5,
                    "recommendation": "建议提供20个以上观测值以获得更可靠的分布特征",
                    "solution": "请收集更多数据样本或调整分析范围",
                },
            )

        # 计算四分位数
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1

        # 计算基本统计量
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        statistics = {
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "count": len(data),
            "variance": float(np.var(data)),
            "range": float(np.max(data) - np.min(data)),
            "coefficient_of_variation": (
                float(std_val / mean_val) if mean_val != 0 else 0
            ),
        }

        # 计算分布形状特征
        skewness = float(np.mean(((data - mean_val) / std_val) ** 3))
        kurtosis = float(np.mean(((data - mean_val) / std_val) ** 4)) - 3  # 超额峰度

        shape_characteristics = {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_symmetric": abs(skewness) < 0.5,
            "distribution_shape": (
                "近似对称"
                if abs(skewness) < 0.5
                else ("右偏(正偏)" if skewness > 0.5 else "左偏(负偏)")
            ),
            "tail_heaviness": (
                "正常"
                if abs(kurtosis) < 0.5
                else ("厚尾" if kurtosis > 0.5 else "薄尾")
            ),
        }

        # 生成分布特征总结
        distribution_summary = f"数据呈{shape_characteristics['distribution_shape']}分布，中位数为{q2:.2f}，四分位距为{iqr:.2f}。数据范围从{np.min(data):.2f}到{np.max(data):.2f}，变异系数为{statistics['coefficient_of_variation']:.3f}。"

        # 生成趋势分析
        if iqr == 0:
            trend_analysis = (
                "数据集中度极高，所有数据点都集中在四分位数范围内，显示出极强的一致性。"
            )
        elif statistics["coefficient_of_variation"] < 0.1:
            trend_analysis = "数据变异性很小，显示出高度的稳定性和一致性。"
        elif statistics["coefficient_of_variation"] < 0.3:
            trend_analysis = "数据变异性适中，整体趋势相对稳定。"
        else:
            trend_analysis = "数据变异性较大，存在明显的波动和分散趋势。"

        # 添加分布形状的趋势描述
        if abs(skewness) > 1:
            trend_analysis += f" 分布明显{shape_characteristics['distribution_shape']}，数据存在不对称的趋势特征。"

        if abs(kurtosis) > 1:
            trend_analysis += f" 分布呈现{shape_characteristics['tail_heaviness']}特征，极值出现的概率{'较高' if kurtosis > 1 else '较低'}。"

        return DistributionResponse(
            quartiles={
                "Q1": float(q1),
                "Q2 (median)": float(q2),
                "Q3": float(q3),
                "IQR": float(iqr),
            },
            statistics=statistics,
            distribution_summary=distribution_summary,
            trend_analysis=trend_analysis,
            shape_characteristics=shape_characteristics,
        )

    except Exception as e:
        logger.error(f"分布趋势分析错误: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"分布趋势分析过程中发生错误: {str(e)}"
        )
