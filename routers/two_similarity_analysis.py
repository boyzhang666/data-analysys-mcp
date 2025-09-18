import logging
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, APIRouter
from scipy import stats
from scipy.spatial.distance import euclidean
from config.config import *

# 全局配置
router = APIRouter()
logger = logging.getLogger("similarity_analysis")


class SimilarityAnalysisRequest(BaseModel):
    """相似度分析请求模型
    
    用于分析两个时间序列数据之间的相似度。
    典型应用场景：比较不同传感器数据的相似性、分析时间序列的匹配程度等。
    """

    data1: List[float] = Field(
        ...,
        description="第一个时间序列数据，用于相似度分析",
        min_items=10,
        example=[1.2, 2.3, 1.8, 4.5, 2.1, 3.2, 1.9, 2.8, 3.5, 6.1],
    )
    data2: List[float] = Field(
        ...,
        description="第二个时间序列数据，与第一个数据分析相似度，长度必须相同",
        min_items=10,
        example=[1.5, 2.1, 2.0, 4.2, 2.3, 3.0, 2.1, 2.9, 3.3, 5.8],
    )
    window_size: Optional[int] = Field(
        default=None,
        description="滑动窗口大小，用于区间相似度分析，默认为数据长度的1/4",
        ge=3,
    )
    normalize: Optional[bool] = Field(
        default=True, description="是否对数据进行标准化处理"
    )


class SimilarityAnalysisResponse(BaseModel):
    """相似度分析响应模型"""

    similarity_score: float = Field(
        ..., description="综合相似度评分，范围0-1，越接近1表示越相似"
    )
    method: str = Field(..., description="使用的相似度分析方法")
    data_length: int = Field(..., description="参与分析的数据点数量")
    interpretation: str = Field(..., description="相似度结果的文字解释")
    best_match_info: Optional[Dict[str, Any]] = Field(None, description="最佳匹配区间信息（可选）")


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """计算DTW距离"""
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), float("inf"))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m] / (n + m)


def calculate_similarity_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """计算多种相似度指标"""
    similarities = {}

    # DTW相似度
    dtw_dist = dtw_distance(x, y)
    similarities["dtw"] = 1 / (1 + dtw_dist)

    # 皮尔逊相关系数
    corr, _ = stats.pearsonr(x, y)
    similarities["pearson"] = abs(corr) if not np.isnan(corr) else 0.0

    # 余弦相似度
    dot_product = np.dot(x, y)
    norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
    if norm_x > 0 and norm_y > 0:
        similarities["cosine"] = abs(dot_product / (norm_x * norm_y))
    else:
        similarities["cosine"] = 0.0

    # 欧氏距离相似度
    eucl_dist = euclidean(x, y)
    max_dist = np.sqrt(len(x)) * (max(np.max(x), np.max(y)) - min(np.min(x), np.min(y)))
    similarities["euclidean"] = 1 - (eucl_dist / max_dist) if max_dist > 0 else 1.0

    return similarities


def sliding_window_analysis(x: np.ndarray, y: np.ndarray, window_size: int) -> tuple:
    """滑动窗口相似度分析"""
    similarities = []
    positions = []

    for i in range(len(x) - window_size + 1):
        window_x = x[i : i + window_size]
        window_y = y[i : i + window_size]

        # 计算窗口内的综合相似度
        window_similarities = calculate_similarity_metrics(window_x, window_y)
        avg_similarity = np.mean(list(window_similarities.values()))

        similarities.append(avg_similarity)
        positions.append(i)

    return similarities, positions


@router.post(
    "/api/similarity_analysis",
    response_model=SimilarityAnalysisResponse,
    operation_id="two_indicators_similarity_analysis",
    tags=["相似度分析"],
)
async def similarity_analysis(request: SimilarityAnalysisRequest):
    """
    分析两个时间序列数据之间的相似度，支持多种相似度计算方法
    
    **参数说明：**
    - **data1**: 第一个时间序列数据，至少包含10个数据点
    - **data2**: 第二个时间序列数据，长度必须与data1相同
    - **window_size**: 滑动窗口大小，用于区间相似度分析
    - **normalize**: 是否对数据进行标准化处理

    **返回结果：**
    - similarity_score: 综合相似度评分（0到1之间）
    - method: 使用的分析方法
    - data_length: 数据点数量
    - interpretation: 相似度解释
    - best_match_info: 最佳匹配区间信息（可选）

    **使用示例：**
    ```json
    {
        "data1": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
        "data2": [1.1, 2.1, 3.1, 4.1, 5.1, 4.1, 3.1, 2.1, 1.1, 2.1],
        "normalize": true
    }
    ```
    """
    try:
        # 使用简化的参数名称
        data1 = request.data1
        data2 = request.data2
        window_size = request.window_size
        normalize = request.normalize

        # 转换为numpy数组
        array1 = np.array(data1)
        array2 = np.array(data2)

        # 检查NaN值
        if np.any(np.isnan(array1)) or np.any(np.isnan(array2)):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "数据包含NaN值",
                    "message": "输入数据中包含NaN值，无法进行相似度分析",
                    "solution": "请清理数据中的NaN值后重新提交",
                },
            )

        # 数据标准化
        if normalize:
            if np.std(array1) > 0:
                array1 = (array1 - np.mean(array1)) / np.std(array1)
            if np.std(array2) > 0:
                array2 = (array2 - np.mean(array2)) / np.std(array2)

        # 计算相似度
        similarity_metrics = calculate_similarity_metrics(array1, array2)
        similarity_score = float(np.mean(list(similarity_metrics.values())))

        # 滑动窗口分析（可选）
        best_match_info = None
        if window_size is None:
            window_size = max(5, len(array1) // 4)
        
        if window_size < len(array1):
            similarity_profile, positions = sliding_window_analysis(array1, array2, window_size)
            if similarity_profile:
                best_idx = np.argmax(similarity_profile)
                best_match_info = {
                    "start_position": int(positions[best_idx]),
                    "end_position": int(positions[best_idx] + window_size - 1),
                    "similarity_score": float(similarity_profile[best_idx]),
                    "window_size": int(window_size),
                }

        # 解释相似度结果
        if similarity_score >= 0.8:
            interpretation = "高度相似"
        elif similarity_score >= 0.6:
            interpretation = "较为相似"
        elif similarity_score >= 0.4:
            interpretation = "中等相似"
        else:
            interpretation = "相似度较低"

        # 构造响应
        return SimilarityAnalysisResponse(
            similarity_score=similarity_score,
            method="综合相似度分析",
            data_length=len(array1),
            interpretation=interpretation,
            best_match_info=best_match_info,
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
                "message": f"相似度分析过程中发生错误: {str(e)}",
                "suggestions": [
                    "检查数据是否包含异常值",
                    "确认数据类型正确",
                    "尝试调整窗口大小参数",
                ],
            },
        )
