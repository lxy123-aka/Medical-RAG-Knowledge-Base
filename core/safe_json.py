"""
safe_json_parse 工具函数（规则6: 统一 JSON 解析，禁止各处重复写 json.loads）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_parse(raw: Any) -> Optional[Union[Dict, List]]:
    """
    统一 JSON 解析工具函数（规则6）

    处理逻辑:
    1. raw 为 dict/list，直接返回
    2. raw 为 str，先 json.loads 解析
    3. 解析失败，用正则提取 JSON 片段再解析
    4. 全部失败返回 None 并记录告警

    Args:
        raw: 待解析的原始数据

    Returns:
        解析后的 dict/list，或 None（解析失败）
    """
    # 1. 已经是 dict/list，直接返回
    if isinstance(raw, (dict, list)):
        return raw

    # 非字符串类型无法解析
    if not isinstance(raw, str):
        logger.warning(f"[safe_json_parse] 不支持的类型: {type(raw)}, 值: {str(raw)[:100]}")
        return None

    raw_str = raw.strip()
    if not raw_str:
        logger.warning("[safe_json_parse] 输入为空字符串")
        return None

    # 2. 直接 json.loads
    try:
        result = json.loads(raw_str)
        if isinstance(result, (dict, list)):
            return result
        # 解析成功但不是 dict/list（如纯数字、字符串）
        logger.warning(f"[safe_json_parse] 解析结果非 dict/list: {type(result)}")
        return None
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. 正则提取 JSON 片段
    # 尝试匹配最外层的 {} 或 []
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 嵌套一层 {}
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # 嵌套一层 []
        r'\{.*\}',  # 贪婪匹配 {}
        r'\[.*\]',  # 贪婪匹配 []
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_str, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, (dict, list)):
                    logger.info(f"[safe_json_parse] 通过正则提取成功解析 JSON")
                    return result
            except (json.JSONDecodeError, ValueError):
                continue

    # 4. 尝试修复常见问题（如单引号、尾部逗号）
    try:
        # 替换单引号为双引号（简单修复）
        fixed = raw_str.replace("'", '"')
        # 移除尾部逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        result = json.loads(fixed)
        if isinstance(result, (dict, list)):
            logger.info("[safe_json_parse] 通过修复常见问题成功解析 JSON")
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 5. 全部失败
    logger.error(
        f"[safe_json_parse] JSON 解析全部失败! "
        f"输入前100字符: {raw_str[:100]}"
    )
    return None


def extract_json_from_markdown(text: str) -> Optional[Union[Dict, List]]:
    """
    从 Markdown 代码块中提取 JSON（规则7: LLM 结构化输出容错）

    支持格式:
    ```json
    {...}
    ```
    或
    ```
    {...}
    ```
    """
    # 匹配 ```json ... ``` 或 ``` ... ```
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        result = safe_json_parse(match)
        if result is not None:
            return result

    # 如果代码块提取失败，尝试整体解析
    return safe_json_parse(text)


def extract_cypher_from_text(text: str) -> Optional[str]:
    """
    从 LLM 输出中提取 Cypher 语句（规则7）

    支持格式:
    ```cypher
    MATCH ...
    ```
    或直接 Cypher 语句
    """
    # 尝试匹配 ```cypher ... ```
    pattern = r'```(?:cypher)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        cypher = match.strip()
        if cypher and any(kw in cypher.upper() for kw in ['MATCH', 'CREATE', 'MERGE', 'RETURN']):
            return cypher

    # 尝试直接提取（查找 MATCH/CREATE/MERGE 开头的行）
    lines = text.split('\n')
    cypher_lines = []
    in_cypher = False

    for line in lines:
        stripped = line.strip()
        if any(kw in stripped.upper() for kw in ['MATCH', 'CREATE', 'MERGE', 'WITH', 'UNWIND']):
            in_cypher = True
        if in_cypher:
            if stripped:
                cypher_lines.append(stripped)
            elif cypher_lines:  # 空行但已有内容，结束
                break

    if cypher_lines:
        return '\n'.join(cypher_lines)

    return None
